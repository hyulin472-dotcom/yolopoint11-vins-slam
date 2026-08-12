#include "yolopoint_cuda_postprocess.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

namespace
{
void checkCuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
}

__global__ void decodeScoresKernel(
    const float *semi,
    float *scores,
    int coarse_width,
    int coarse_height)
{
    const int coarse_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int coarse_area = coarse_width * coarse_height;
    if (coarse_index >= coarse_area)
        return;

    float maximum = -FLT_MAX;
    for (int channel = 0; channel < 65; ++channel)
        maximum = fmaxf(maximum, semi[channel * coarse_area + coarse_index]);

    float sum = 0.0f;
    float probabilities[64];
    for (int channel = 0; channel < 65; ++channel)
    {
        const float value =
            expf(semi[channel * coarse_area + coarse_index] - maximum);
        sum += value;
        if (channel < 64)
            probabilities[channel] = value;
    }

    const int coarse_y = coarse_index / coarse_width;
    const int coarse_x = coarse_index - coarse_y * coarse_width;
    const int image_width = coarse_width * 8;
    for (int channel = 0; channel < 64; ++channel)
    {
        const int y = coarse_y * 8 + channel / 8;
        const int x = coarse_x * 8 + channel % 8;
        scores[y * image_width + x] = probabilities[channel] / sum;
    }
}

__global__ void initialMaximaKernel(
    const float *scores,
    unsigned char *maxima,
    int width,
    int height,
    int radius)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int area = width * height;
    if (index >= area)
        return;
    const int y = index / width;
    const int x = index - y * width;
    const float value = scores[index];
    float pooled = -FLT_MAX;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        const int sy = y + dy;
        if (sy < 0 || sy >= height)
            continue;
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int sx = x + dx;
            if (sx >= 0 && sx < width)
                pooled = fmaxf(pooled, scores[sy * width + sx]);
        }
    }
    maxima[index] = value == pooled ? 1 : 0;
}

__global__ void suppressionKernel(
    const unsigned char *maxima,
    unsigned char *suppressed,
    int width,
    int height,
    int radius)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int area = width * height;
    if (index >= area)
        return;
    const int y = index / width;
    const int x = index - y * width;
    unsigned char found = 0;
    for (int dy = -radius; dy <= radius && !found; ++dy)
    {
        const int sy = y + dy;
        if (sy < 0 || sy >= height)
            continue;
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int sx = x + dx;
            if (sx >= 0 && sx < width && maxima[sy * width + sx])
            {
                found = 1;
                break;
            }
        }
    }
    suppressed[index] = found;
}

__global__ void remainingScoresKernel(
    const float *scores,
    const unsigned char *suppressed,
    float *remaining,
    int area)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < area)
        remaining[index] = suppressed[index] ? 0.0f : scores[index];
}

__global__ void updateMaximaKernel(
    const float *remaining,
    const unsigned char *suppressed,
    unsigned char *maxima,
    int width,
    int height,
    int radius)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int area = width * height;
    if (index >= area || suppressed[index])
        return;
    const int y = index / width;
    const int x = index - y * width;
    float pooled = -FLT_MAX;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        const int sy = y + dy;
        if (sy < 0 || sy >= height)
            continue;
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int sx = x + dx;
            if (sx >= 0 && sx < width)
                pooled = fmaxf(pooled, remaining[sy * width + sx]);
        }
    }
    if (remaining[index] == pooled)
        maxima[index] = 1;
}

__global__ void collectCandidatesKernel(
    const float *scores,
    const unsigned char *maxima,
    float threshold,
    int border,
    int width,
    int height,
    std::uint64_t *candidate_keys,
    int *candidate_count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int area = width * height;
    if (index >= area)
        return;
    const int y = index / width;
    const int x = index - y * width;
    if (x < border || x >= width - border ||
        y < border || y >= height - border ||
        !maxima[index] || scores[index] <= threshold)
        return;
    const int output = atomicAdd(candidate_count, 1);
    const std::uint64_t score_bits = __float_as_uint(scores[index]);
    const std::uint64_t index_tiebreak =
        0xffffffffULL - static_cast<std::uint32_t>(index);
    candidate_keys[output] = (score_bits << 32) | index_tiebreak;
}

__global__ void sampleDescriptorsKernel(
    const float *dense_descriptors,
    const std::uint64_t *candidate_keys,
    float *output_descriptors,
    int keypoint_count,
    int image_width,
    int descriptor_width,
    int descriptor_height,
    int descriptor_dimension)
{
    const int keypoint = blockIdx.x;
    const int channel = threadIdx.x;
    if (keypoint >= keypoint_count || channel >= descriptor_dimension)
        return;

    const int packed = static_cast<int>(
        0xffffffffULL - (candidate_keys[keypoint] & 0xffffffffULL));
    const int pixel_y = packed / image_width;
    const int pixel_x = packed - pixel_y * image_width;
    const float descriptor_x = (pixel_x + 0.5f) / 8.0f - 0.5f;
    const float descriptor_y = (pixel_y + 0.5f) / 8.0f - 0.5f;
    const int x0 = static_cast<int>(floorf(descriptor_x));
    const int y0 = static_cast<int>(floorf(descriptor_y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float wx = descriptor_x - x0;
    const float wy = descriptor_y - y0;
    const int descriptor_area = descriptor_width * descriptor_height;

    auto sample = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= descriptor_width || y >= descriptor_height)
            return 0.0f;
        return dense_descriptors[
            channel * descriptor_area + y * descriptor_width + x];
    };
    const float top = (1.0f - wx) * sample(x0, y0) + wx * sample(x1, y0);
    const float bottom =
        (1.0f - wx) * sample(x0, y1) + wx * sample(x1, y1);
    const float value = (1.0f - wy) * top + wy * bottom;

    __shared__ float squared_norm[128];
    squared_norm[channel] = value * value;
    __syncthreads();
    for (int stride = descriptor_dimension / 2; stride > 0; stride >>= 1)
    {
        if (channel < stride)
            squared_norm[channel] += squared_norm[channel + stride];
        __syncthreads();
    }
    const float inverse_norm = rsqrtf(fmaxf(squared_norm[0], 1e-12f));
    output_descriptors[
        keypoint * descriptor_dimension + channel] = value * inverse_norm;
}

__global__ void meanDescriptorKernel(
    const float *dense_descriptors,
    float *global_descriptor,
    int descriptor_area)
{
    const int channel = blockIdx.x;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (int index = threadIdx.x; index < descriptor_area; index += blockDim.x)
        sum += dense_descriptors[channel * descriptor_area + index];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        global_descriptor[channel] = sums[0] / descriptor_area;
}

__global__ void normalizeGlobalDescriptorKernel(float *descriptor)
{
    const int channel = threadIdx.x;
    __shared__ float squared_norm[128];
    squared_norm[channel] = descriptor[channel] * descriptor[channel];
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (channel < stride)
            squared_norm[channel] += squared_norm[channel + stride];
        __syncthreads();
    }
    descriptor[channel] *= rsqrtf(fmaxf(squared_norm[0], 1e-12f));
}
}  // namespace

struct YOLOPointCudaPostprocessor::Impl
{
    int image_width;
    int image_height;
    int coarse_width;
    int coarse_height;
    int descriptor_dimension;
    int max_keypoints;
    int image_area;
    int coarse_area;

    float *semi = nullptr;
    float *dense_descriptors = nullptr;
    float *scores = nullptr;
    float *remaining_scores = nullptr;
    unsigned char *maxima = nullptr;
    unsigned char *suppressed = nullptr;
    std::uint64_t *candidate_keys = nullptr;
    int *candidate_count = nullptr;
    float *selected_descriptors = nullptr;
    float *global_descriptor = nullptr;

    std::vector<std::uint64_t> host_candidate_keys;

    Impl(int width, int height, int descriptor_dim, int maximum_keypoints)
        : image_width(width),
          image_height(height),
          coarse_width(width / 8),
          coarse_height(height / 8),
          descriptor_dimension(descriptor_dim),
          max_keypoints(maximum_keypoints),
          image_area(width * height),
          coarse_area((width / 8) * (height / 8)),
          host_candidate_keys(maximum_keypoints)
    {
        if (width % 8 != 0 || height % 8 != 0 || descriptor_dim != 128)
            throw std::runtime_error("unsupported YOLOPoint CUDA output shape");
        checkCuda(cudaMalloc(&semi, 65ULL * coarse_area * sizeof(float)),
                  "cudaMalloc semi");
        checkCuda(cudaMalloc(
                      &dense_descriptors,
                      static_cast<size_t>(descriptor_dimension) *
                          coarse_area * sizeof(float)),
                  "cudaMalloc dense descriptors");
        checkCuda(cudaMalloc(&scores, image_area * sizeof(float)),
                  "cudaMalloc scores");
        checkCuda(cudaMalloc(&remaining_scores, image_area * sizeof(float)),
                  "cudaMalloc remaining scores");
        checkCuda(cudaMalloc(&maxima, image_area * sizeof(unsigned char)),
                  "cudaMalloc maxima");
        checkCuda(cudaMalloc(&suppressed, image_area * sizeof(unsigned char)),
                  "cudaMalloc suppressed");
        checkCuda(cudaMalloc(
                      &candidate_keys,
                      static_cast<size_t>(image_area) * sizeof(std::uint64_t)),
                  "cudaMalloc candidate keys");
        checkCuda(cudaMalloc(&candidate_count, sizeof(int)),
                  "cudaMalloc candidate count");
        checkCuda(cudaMalloc(
                      &selected_descriptors,
                      static_cast<size_t>(max_keypoints) *
                          descriptor_dimension * sizeof(float)),
                  "cudaMalloc selected descriptors");
        checkCuda(cudaMalloc(
                      &global_descriptor,
                      descriptor_dimension * sizeof(float)),
                  "cudaMalloc global descriptor");
    }

    ~Impl()
    {
        cudaFree(global_descriptor);
        cudaFree(selected_descriptors);
        cudaFree(candidate_count);
        cudaFree(candidate_keys);
        cudaFree(suppressed);
        cudaFree(maxima);
        cudaFree(remaining_scores);
        cudaFree(scores);
        cudaFree(dense_descriptors);
        cudaFree(semi);
    }
};

YOLOPointCudaPostprocessor::YOLOPointCudaPostprocessor(
    int image_width,
    int image_height,
    int descriptor_dimension,
    int max_keypoints)
    : impl_(new Impl(
          image_width, image_height, descriptor_dimension, max_keypoints))
{
}

YOLOPointCudaPostprocessor::~YOLOPointCudaPostprocessor() = default;

float *YOLOPointCudaPostprocessor::semiDeviceData()
{
    return impl_->semi;
}

float *YOLOPointCudaPostprocessor::denseDescriptorDeviceData()
{
    return impl_->dense_descriptors;
}

size_t YOLOPointCudaPostprocessor::semiElementCount() const
{
    return 65ULL * impl_->coarse_area;
}

size_t YOLOPointCudaPostprocessor::denseDescriptorElementCount() const
{
    return static_cast<size_t>(impl_->descriptor_dimension) * impl_->coarse_area;
}

void YOLOPointCudaPostprocessor::globalDescriptor(cv::Mat &descriptor)
{
    meanDescriptorKernel<<<impl_->descriptor_dimension, 256>>>(
        impl_->dense_descriptors, impl_->global_descriptor, impl_->coarse_area);
    normalizeGlobalDescriptorKernel<<<1, impl_->descriptor_dimension>>>(
        impl_->global_descriptor);
    checkCuda(cudaGetLastError(), "YOLOPoint CUDA global descriptor kernels");
    descriptor.create(1, impl_->descriptor_dimension, CV_32FC1);
    checkCuda(cudaMemcpy(
                  descriptor.ptr<float>(), impl_->global_descriptor,
                  impl_->descriptor_dimension * sizeof(float),
                  cudaMemcpyDeviceToHost),
              "copy global descriptor");
}

void YOLOPointCudaPostprocessor::process(
    float score_threshold,
    int nms_radius,
    int remove_borders,
    float scale_x,
    float scale_y,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat &descriptors)
{
    const int threads = 256;
    const int coarse_blocks = (impl_->coarse_area + threads - 1) / threads;
    const int image_blocks = (impl_->image_area + threads - 1) / threads;
    decodeScoresKernel<<<coarse_blocks, threads>>>(
        impl_->semi, impl_->scores,
        impl_->coarse_width, impl_->coarse_height);
    initialMaximaKernel<<<image_blocks, threads>>>(
        impl_->scores, impl_->maxima,
        impl_->image_width, impl_->image_height, nms_radius);
    for (int iteration = 0; iteration < 2; ++iteration)
    {
        suppressionKernel<<<image_blocks, threads>>>(
            impl_->maxima, impl_->suppressed,
            impl_->image_width, impl_->image_height, nms_radius);
        remainingScoresKernel<<<image_blocks, threads>>>(
            impl_->scores, impl_->suppressed,
            impl_->remaining_scores, impl_->image_area);
        updateMaximaKernel<<<image_blocks, threads>>>(
            impl_->remaining_scores, impl_->suppressed, impl_->maxima,
            impl_->image_width, impl_->image_height, nms_radius);
    }

    checkCuda(cudaMemset(impl_->candidate_count, 0, sizeof(int)),
              "cudaMemset candidate count");
    collectCandidatesKernel<<<image_blocks, threads>>>(
        impl_->scores, impl_->maxima, score_threshold, remove_borders,
        impl_->image_width, impl_->image_height,
        impl_->candidate_keys, impl_->candidate_count);
    checkCuda(cudaGetLastError(), "YOLOPoint CUDA candidate kernels");

    int all_candidate_count = 0;
    checkCuda(cudaMemcpy(
                  &all_candidate_count, impl_->candidate_count, sizeof(int),
                  cudaMemcpyDeviceToHost),
              "copy candidate count");
    const int candidate_count =
        std::min(all_candidate_count, impl_->max_keypoints);
    if (candidate_count <= 0)
    {
        keypoints.clear();
        descriptors.release();
        return;
    }

    thrust::device_ptr<std::uint64_t> key_begin(impl_->candidate_keys);
    thrust::sort(
        key_begin, key_begin + all_candidate_count,
        thrust::greater<std::uint64_t>());

    checkCuda(cudaMemcpy(
                  impl_->host_candidate_keys.data(), impl_->candidate_keys,
                  candidate_count * sizeof(std::uint64_t),
                  cudaMemcpyDeviceToHost),
              "copy candidate keys");

    sampleDescriptorsKernel<<<candidate_count, impl_->descriptor_dimension>>>(
        impl_->dense_descriptors, impl_->candidate_keys,
        impl_->selected_descriptors, candidate_count,
        impl_->image_width, impl_->coarse_width, impl_->coarse_height,
        impl_->descriptor_dimension);
    checkCuda(cudaGetLastError(), "YOLOPoint CUDA descriptor kernel");

    descriptors.create(
        candidate_count, impl_->descriptor_dimension, CV_32FC1);
    checkCuda(cudaMemcpy(
                  descriptors.ptr<float>(), impl_->selected_descriptors,
                  static_cast<size_t>(candidate_count) *
                      impl_->descriptor_dimension * sizeof(float),
                  cudaMemcpyDeviceToHost),
              "copy selected descriptors");

    keypoints.resize(candidate_count);
    for (int index = 0; index < candidate_count; ++index)
    {
        const std::uint64_t key = impl_->host_candidate_keys[index];
        const int packed = static_cast<int>(
            0xffffffffULL - (key & 0xffffffffULL));
        const std::uint32_t score_bits =
            static_cast<std::uint32_t>(key >> 32);
        float score = 0.0f;
        std::memcpy(&score, &score_bits, sizeof(float));
        const int y = packed / impl_->image_width;
        const int x = packed - y * impl_->image_width;
        cv::KeyPoint &keypoint = keypoints[index];
        keypoint.pt = cv::Point2f(
            (x + 0.5f) * scale_x, (y + 0.5f) * scale_y);
        keypoint.response = score;
        keypoint.size = 1.0f;
        keypoint.class_id = packed;
    }
}

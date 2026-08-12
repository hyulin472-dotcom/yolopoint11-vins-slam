#include "yolopoint_cuda_postprocess.h"

#include <stdexcept>

struct YOLOPointCudaPostprocessor::Impl {};

YOLOPointCudaPostprocessor::YOLOPointCudaPostprocessor(
    int, int, int, int)
{
    throw std::runtime_error(
        "CUDA postprocessing is unavailable in this build; use the CPU backend");
}

YOLOPointCudaPostprocessor::~YOLOPointCudaPostprocessor() = default;
float *YOLOPointCudaPostprocessor::semiDeviceData() { return nullptr; }
float *YOLOPointCudaPostprocessor::denseDescriptorDeviceData() { return nullptr; }
size_t YOLOPointCudaPostprocessor::semiElementCount() const { return 0; }
size_t YOLOPointCudaPostprocessor::denseDescriptorElementCount() const { return 0; }
void YOLOPointCudaPostprocessor::globalDescriptor(cv::Mat &) {}
void YOLOPointCudaPostprocessor::process(
    float, int, int, float, float,
    std::vector<cv::KeyPoint> &, cv::Mat &)
{
    throw std::runtime_error("CUDA postprocessing is unavailable in this build");
}

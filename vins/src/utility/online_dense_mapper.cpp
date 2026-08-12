/*******************************************************
 * Online keyframe dense stereo mapper for RViz visualization.
 *******************************************************/

#include "online_dense_mapper.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>

#include <opencv2/ccalib/omnidir.hpp>
#include <opencv2/core/eigen.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include "../estimator/parameters.h"

namespace
{
// The deployed single-graph Fast-FoundationStereo ONNX has its internal
// normalization stripped. It expects RGB float input normalized with the
// ImageNet statistics expressed in the original [0, 255] pixel scale.
constexpr float kFoundationImageNetMean[3] = {123.675f, 116.280f, 103.530f};
constexpr float kFoundationImageNetStd[3] = {58.395f, 57.120f, 57.375f};

cv::Mat readCameraMatrix(const cv::FileStorage &fs)
{
    cv::FileNode proj = fs["projection_parameters"];
    double fx = static_cast<double>(proj["gamma1"]);
    double fy = static_cast<double>(proj["gamma2"]);
    double cx = static_cast<double>(proj["u0"]);
    double cy = static_cast<double>(proj["v0"]);
    if (fx == 0.0 && fy == 0.0)
    {
        fx = static_cast<double>(proj["fx"]);
        fy = static_cast<double>(proj["fy"]);
        cx = static_cast<double>(proj["cx"]);
        cy = static_cast<double>(proj["cy"]);
    }
    cv::Mat k = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
                 0.0, fy, cy,
                 0.0, 0.0, 1.0);
    return k;
}

cv::Mat readDistortion(const cv::FileStorage &fs)
{
    cv::FileNode dist = fs["distortion_parameters"];
    return (cv::Mat_<double>(1, 4) << static_cast<double>(dist["k1"]), static_cast<double>(dist["k2"]),
            static_cast<double>(dist["p1"]), static_cast<double>(dist["p2"]));
}

double readXi(const cv::FileStorage &fs)
{
    if (fs["mirror_parameters"].empty())
        return 0.0;
    return static_cast<double>(fs["mirror_parameters"]["xi"]);
}

std::string readCameraModel(const cv::FileStorage &fs)
{
    std::string model_type;
    fs["model_type"] >> model_type;
    return model_type;
}

uint32_t grayToRgb(uint8_t gray)
{
    return (static_cast<uint32_t>(gray) << 16) | (static_cast<uint32_t>(gray) << 8) | static_cast<uint32_t>(gray);
}

uint32_t bgrToRgb(const cv::Vec3b &bgr)
{
    return (static_cast<uint32_t>(bgr[2]) << 16) | (static_cast<uint32_t>(bgr[1]) << 8) | static_cast<uint32_t>(bgr[0]);
}

void ensureDirectory(const std::string &path)
{
    if (path.empty())
        return;
    std::string current;
    if (path[0] == '/')
        current = "/";
    size_t start = path[0] == '/' ? 1 : 0;
    while (start <= path.size())
    {
        size_t end = path.find('/', start);
        std::string part = path.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!part.empty())
        {
            if (!current.empty() && current.back() != '/')
                current += "/";
            current += part;
            mkdir(current.c_str(), 0755);
        }
        if (end == std::string::npos)
            break;
        start = end + 1;
    }
}

int64_t denseGridKey(int64_t x, int64_t y, int64_t z)
{
    return ((x & 0x1fffff) << 42) ^ ((y & 0x1fffff) << 21) ^ (z & 0x1fffff);
}

Eigen::Vector3i denseGridIndex(const Eigen::Vector3f &p, double cell)
{
    const double inv = 1.0 / std::max(cell, 1e-6);
    return Eigen::Vector3i(static_cast<int>(std::floor(p.x() * inv)),
                           static_cast<int>(std::floor(p.y() * inv)),
                           static_cast<int>(std::floor(p.z() * inv)));
}

double logOdds(double probability)
{
    const double p = std::min(0.99, std::max(0.01, probability));
    return std::log(p / (1.0 - p));
}

void voxelFilterLocalCloud(std::vector<Eigen::Vector3f> &points, std::vector<uint32_t> &colors, double voxel_size)
{
    struct Accum
    {
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        uint64_t r = 0;
        uint64_t g = 0;
        uint64_t b = 0;
        uint32_t count = 0;
    };

    std::unordered_map<int64_t, Accum> voxels;
    voxels.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i)
    {
        Eigen::Vector3i idx = denseGridIndex(points[i], voxel_size);
        Accum &acc = voxels[denseGridKey(idx.x(), idx.y(), idx.z())];
        acc.sum += points[i];
        acc.r += (colors[i] >> 16) & 0xff;
        acc.g += (colors[i] >> 8) & 0xff;
        acc.b += colors[i] & 0xff;
        ++acc.count;
    }

    points.clear();
    colors.clear();
    points.reserve(voxels.size());
    colors.reserve(voxels.size());
    for (const auto &kv : voxels)
    {
        const Accum &acc = kv.second;
        points.push_back(acc.sum / static_cast<float>(acc.count));
        uint32_t r = static_cast<uint32_t>(acc.r / acc.count);
        uint32_t g = static_cast<uint32_t>(acc.g / acc.count);
        uint32_t b = static_cast<uint32_t>(acc.b / acc.count);
        colors.push_back((r << 16) | (g << 8) | b);
    }
}

void radiusFilterLocalCloud(std::vector<Eigen::Vector3f> &points,
                            std::vector<uint32_t> &colors,
                            double radius,
                            int min_neighbors)
{
    if (points.empty())
        return;

    const double radius2 = radius * radius;
    std::unordered_map<int64_t, std::vector<int>> grid;
    grid.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i)
    {
        Eigen::Vector3i idx = denseGridIndex(points[i], radius);
        grid[denseGridKey(idx.x(), idx.y(), idx.z())].push_back(static_cast<int>(i));
    }

    std::vector<Eigen::Vector3f> kept_points;
    std::vector<uint32_t> kept_colors;
    kept_points.reserve(points.size());
    kept_colors.reserve(colors.size());
    for (size_t i = 0; i < points.size(); ++i)
    {
        Eigen::Vector3i idx = denseGridIndex(points[i], radius);
        int neighbors = 0;
        for (int dx = -1; dx <= 1 && neighbors < min_neighbors; ++dx)
        {
            for (int dy = -1; dy <= 1 && neighbors < min_neighbors; ++dy)
            {
                for (int dz = -1; dz <= 1 && neighbors < min_neighbors; ++dz)
                {
                    auto it = grid.find(denseGridKey(idx.x() + dx, idx.y() + dy, idx.z() + dz));
                    if (it == grid.end())
                        continue;
                    for (int j : it->second)
                    {
                        if (j == static_cast<int>(i))
                            continue;
                        if ((points[i] - points[j]).squaredNorm() <= radius2)
                        {
                            ++neighbors;
                            if (neighbors >= min_neighbors)
                                break;
                        }
                    }
                }
            }
        }
        if (neighbors >= min_neighbors)
        {
            kept_points.push_back(points[i]);
            kept_colors.push_back(colors[i]);
        }
    }

    points.swap(kept_points);
    colors.swap(kept_colors);
}
}

void OnlineDenseMapper::init(rclcpp::Node::SharedPtr node)
{
    if (!ONLINE_DENSE_MAPPING_ENABLE)
        return;

    pub_cloud_ = node->create_publisher<sensor_msgs::msg::PointCloud2>(ONLINE_DENSE_MAPPING_TOPIC, 1);
    initialized_ = buildRectifier();
    initialized_ = initialized_ && initFoundationStereo();
    if (!initialized_)
    {
        ROS_WARN("online dense mapping failed to initialize; disable online dense map publisher");
        return;
    }

    ROS_INFO("online FoundationStereo dense mapping enabled: topic %s, keyframe stride %d, max points/keyframe %d",
             ONLINE_DENSE_MAPPING_TOPIC.c_str(),
             ONLINE_DENSE_MAPPING_KEYFRAME_STRIDE,
             ONLINE_DENSE_MAPPING_MAX_POINTS_PER_KEYFRAME);
}

void OnlineDenseMapper::reset()
{
    keyframe_counter_ = 0;
    points_.clear();
    colors_.clear();
    occupied_voxels_.clear();
    submaps_.clear();
    submap_index_by_stamp_.clear();
}

void OnlineDenseMapper::shutdown()
{
    initialized_ = false;
    reset();
    ort_session_.reset();
    pub_cloud_.reset();
}

bool OnlineDenseMapper::buildRectifier()
{
    if (CAM_NAMES.size() < 2 || RIC.size() < 2 || TIC.size() < 2)
    {
        ROS_WARN("online dense mapping requires stereo camera calibration");
        return false;
    }

    cv::FileStorage fs0(CAM_NAMES[0], cv::FileStorage::READ);
    cv::FileStorage fs1(CAM_NAMES[1], cv::FileStorage::READ);
    if (!fs0.isOpened() || !fs1.isOpened())
    {
        ROS_WARN("online dense mapping cannot open camera yaml files");
        return false;
    }

    cv::Mat k0 = readCameraMatrix(fs0);
    cv::Mat k1 = readCameraMatrix(fs1);
    cv::Mat d0 = readDistortion(fs0);
    cv::Mat d1 = readDistortion(fs1);
    std::string model0 = readCameraModel(fs0);
    std::string model1 = readCameraModel(fs1);

    Eigen::Matrix3d r01 = RIC[0].transpose() * RIC[1];
    Eigen::Vector3d t01 = RIC[0].transpose() * (TIC[1] - TIC[0]);
    baseline_ = t01.norm();
    if (baseline_ <= 1e-6)
    {
        ROS_WARN("online dense mapping invalid stereo baseline");
        return false;
    }

    cv::Mat r_cv, t_cv;
    cv::eigen2cv(r01, r_cv);
    cv::eigen2cv(t01, t_cv);

    fx_ = ONLINE_DENSE_MAPPING_RECTIFIED_FOCAL;
    cx_ = COL * 0.5;
    cy_ = ROW * 0.5;
    cv::Size size(COL, ROW);
    if (model0 == "PINHOLE" && model1 == "PINHOLE")
    {
        cv::Mat r1, r2, p1, p2, q;
        cv::stereoRectify(k0, d0, k1, d1, size, r_cv, t_cv, r1, r2, p1, p2, q,
                          cv::CALIB_ZERO_DISPARITY, 0, size);
        if (p1.rows == 3 && p1.cols == 4)
        {
            fx_ = p1.at<double>(0, 0);
            cx_ = p1.at<double>(0, 2);
            cy_ = p1.at<double>(1, 2);
        }
        cv::initUndistortRectifyMap(k0, d0, r1, p1, size, CV_32FC1, map00_, map01_);
        cv::initUndistortRectifyMap(k1, d1, r2, p2, size, CV_32FC1, map10_, map11_);
    }
    else
    {
        Eigen::Matrix3d r10 = r01.transpose();
        Eigen::Vector3d t10 = -r10 * t01;
        cv::eigen2cv(r10, r_cv);
        cv::eigen2cv(t10, t_cv);
        cv::Mat xi0 = (cv::Mat_<double>(1, 1) << readXi(fs0));
        cv::Mat xi1 = (cv::Mat_<double>(1, 1) << readXi(fs1));
        cv::Mat r1, r2;
        cv::omnidir::stereoRectify(r_cv, t_cv, r1, r2);
        cv::Mat p = (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fx_, cy_, 0.0, 0.0, 1.0);
        cv::omnidir::initUndistortRectifyMap(k0, d0, xi0, r1, p, size, CV_32FC1, map00_, map01_, cv::omnidir::RECTIFY_PERSPECTIVE);
        cv::omnidir::initUndistortRectifyMap(k1, d1, xi1, r2, p, size, CV_32FC1, map10_, map11_, cv::omnidir::RECTIFY_PERSPECTIVE);
    }
    ROS_INFO("online dense mapping rectified fx %.3f baseline %.6f model %s/%s", fx_, baseline_, model0.c_str(), model1.c_str());
    return true;
}

bool OnlineDenseMapper::initFoundationStereo()
{
    try
    {
        input_shape_ = {1, 3, ONLINE_DENSE_MAPPING_FOUNDATION_INPUT_HEIGHT, ONLINE_DENSE_MAPPING_FOUNDATION_INPUT_WIDTH};
        ort_session_options_ = Ort::SessionOptions();
        ort_session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        ort_session_options_.SetIntraOpNumThreads(1);
        if (ONLINE_DENSE_MAPPING_FOUNDATION_USE_TENSORRT)
        {
            ensureDirectory(ONLINE_DENSE_MAPPING_FOUNDATION_TRT_CACHE_PATH);
            OrtTensorRTProviderOptions trt_options{};
            trt_options.device_id = 0;
            trt_options.trt_max_partition_iterations = 1000;
            trt_options.trt_fp16_enable = 1;
            trt_options.trt_min_subgraph_size = 1;
            trt_options.trt_max_workspace_size = 1ULL << 32;
            trt_options.trt_engine_cache_enable = 1;
            trt_options.trt_engine_cache_path = ONLINE_DENSE_MAPPING_FOUNDATION_TRT_CACHE_PATH.c_str();
            ort_session_options_.AppendExecutionProvider_TensorRT(trt_options);

            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            cuda_options.gpu_mem_limit = 0;
            cuda_options.arena_extend_strategy = 1;
            cuda_options.do_copy_in_default_stream = 1;
            ort_session_options_.AppendExecutionProvider_CUDA(cuda_options);
            ROS_INFO("online dense mapping uses TensorRTExecutionProvider + CUDAExecutionProvider");
        }
        else if (ONLINE_DENSE_MAPPING_FOUNDATION_USE_CUDA)
        {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            cuda_options.gpu_mem_limit = 0;
            cuda_options.arena_extend_strategy = 1;
            cuda_options.do_copy_in_default_stream = 1;
            ort_session_options_.AppendExecutionProvider_CUDA(cuda_options);
            ROS_INFO("online dense mapping uses CUDAExecutionProvider");
        }
        else
        {
            ROS_INFO("online dense mapping uses CPUExecutionProvider");
        }

        const std::string model_path = ONLINE_DENSE_MAPPING_FOUNDATION_MODEL_PATH;
        try
        {
            ort_session_.reset(new Ort::Session(ort_env_, model_path.c_str(), ort_session_options_));
        }
        catch (const std::exception &e)
        {
            if (!ONLINE_DENSE_MAPPING_FOUNDATION_USE_TENSORRT && !ONLINE_DENSE_MAPPING_FOUNDATION_USE_CUDA)
                throw;
            ROS_WARN("online dense mapping GPU session failed: %s", e.what());
            ROS_WARN("online dense mapping fallback to CPUExecutionProvider");
            ort_session_options_ = Ort::SessionOptions();
            ort_session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            ort_session_options_.SetIntraOpNumThreads(1);
            ort_session_.reset(new Ort::Session(ort_env_, model_path.c_str(), ort_session_options_));
        }

        input_name_storage_.clear();
        output_name_storage_.clear();
        input_names_.clear();
        output_names_.clear();
        for (size_t i = 0; i < ort_session_->GetInputCount(); ++i)
        {
            auto name = ort_session_->GetInputNameAllocated(i, ort_allocator_);
            input_name_storage_.emplace_back(name.get());
        }
        for (size_t i = 0; i < ort_session_->GetOutputCount(); ++i)
        {
            auto name = ort_session_->GetOutputNameAllocated(i, ort_allocator_);
            output_name_storage_.emplace_back(name.get());
        }
        for (const auto &name : input_name_storage_)
            input_names_.push_back(name.c_str());
        for (const auto &name : output_name_storage_)
            output_names_.push_back(name.c_str());
        if (input_names_.size() < 2 || output_names_.empty())
            throw std::runtime_error("FoundationStereo ONNX expects two inputs and at least one output");
        ROS_INFO("online dense mapping model: %s", model_path.c_str());
        return true;
    }
    catch (const std::exception &e)
    {
        ROS_WARN("online dense mapping FoundationStereo init failed: %s", e.what());
        return false;
    }
}

std::vector<float> OnlineDenseMapper::imageToTensor(const cv::Mat &image) const
{
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(ONLINE_DENSE_MAPPING_FOUNDATION_INPUT_WIDTH,
                                        ONLINE_DENSE_MAPPING_FOUNDATION_INPUT_HEIGHT),
               0, 0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);

    const int h = rgb.rows;
    const int w = rgb.cols;
    std::vector<float> tensor(static_cast<size_t>(3) * h * w);
    for (int y = 0; y < h; ++y)
    {
        const cv::Vec3b *row = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                const float pixel = static_cast<float>(row[x][c]);
                tensor[(static_cast<size_t>(c) * h + y) * w + x] =
                    (pixel - kFoundationImageNetMean[c]) / kFoundationImageNetStd[c];
            }
        }
    }
    return tensor;
}

bool OnlineDenseMapper::inferFoundationStereo(const cv::Mat &left_rect, const cv::Mat &right_rect, cv::Mat &disparity)
{
    if (!ort_session_)
        return false;
    try
    {
        std::vector<float> left_tensor = imageToTensor(left_rect);
        std::vector<float> right_tensor = imageToTensor(right_rect);
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info, left_tensor.data(), left_tensor.size(),
                                                            input_shape_.data(), input_shape_.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info, right_tensor.data(), right_tensor.size(),
                                                            input_shape_.data(), input_shape_.size()));
        auto outputs = ort_session_->Run(Ort::RunOptions{nullptr},
                                         input_names_.data(), inputs.data(), inputs.size(),
                                         output_names_.data(), output_names_.size());
        if (outputs.empty())
            return false;
        const auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (output_shape.size() < 4)
            return false;
        const int out_h = static_cast<int>(output_shape[2]);
        const int out_w = static_cast<int>(output_shape[3]);
        const float *data = outputs[0].GetTensorData<float>();
        cv::Mat small(out_h, out_w, CV_32F, const_cast<float *>(data));
        cv::resize(small, disparity, cv::Size(COL, ROW), 0, 0, cv::INTER_LINEAR);
        disparity *= static_cast<float>(COL) / static_cast<float>(std::max(1, out_w));
        return true;
    }
    catch (const std::exception &e)
    {
        ROS_WARN("online dense mapping FoundationStereo inference failed: %s", e.what());
        return false;
    }
}

int64_t OnlineDenseMapper::voxelKey(const Eigen::Vector3f &p) const
{
    const double inv = 1.0 / std::max(ONLINE_DENSE_MAPPING_VOXEL_SIZE, 1e-6);
    const int64_t x = static_cast<int64_t>(std::floor(p.x() * inv));
    const int64_t y = static_cast<int64_t>(std::floor(p.y() * inv));
    const int64_t z = static_cast<int64_t>(std::floor(p.z() * inv));
    return ((x & 0x1fffff) << 42) ^ ((y & 0x1fffff) << 21) ^ (z & 0x1fffff);
}

int64_t OnlineDenseMapper::stampKey(const std_msgs::msg::Header &header) const
{
    return static_cast<int64_t>(header.stamp.sec) * 1000000000LL +
           static_cast<int64_t>(header.stamp.nanosec);
}

void OnlineDenseMapper::processKeyframe(const cv::Mat &left,
                                        const cv::Mat &right,
                                        const cv::Mat &left_color,
                                        const Eigen::Vector3d &p_wb,
                                        const Eigen::Matrix3d &r_wb,
                                        const std_msgs::msg::Header &header)
{
    static int skipped_frames = 0;
    if (!ONLINE_DENSE_MAPPING_ENABLE || !initialized_ || left.empty() || right.empty() || !ort_session_)
    {
        if (++skipped_frames % 20 == 1)
        {
            ROS_WARN("online dense skip: enable %d initialized %d left_empty %d right_empty %d session %d",
                     ONLINE_DENSE_MAPPING_ENABLE,
                     initialized_,
                     left.empty(),
                     right.empty(),
                     ort_session_ ? 1 : 0);
        }
        return;
    }

    const int64_t submap_stamp_key = stampKey(header);
    if (ONLINE_DENSE_MAPPING_USE_SUBMAP_MANAGER && ONLINE_DENSE_MAPPING_USE_POSE_UPDATE_REASSEMBLY)
    {
        auto it = submap_index_by_stamp_.find(submap_stamp_key);
        if (it != submap_index_by_stamp_.end() && it->second < submaps_.size())
        {
            DenseSubmap &submap = submaps_[it->second];
            submap.p_wb = p_wb;
            submap.r_wb = r_wb;
            ROS_INFO("online dense submap pose updated: stamp %ld index %zu", submap_stamp_key, it->second);
            publishSubmaps(header);
            return;
        }
    }

    if (++keyframe_counter_ % std::max(1, ONLINE_DENSE_MAPPING_KEYFRAME_STRIDE) != 0)
        return;
    if (!ONLINE_DENSE_MAPPING_USE_SUBMAP_MANAGER &&
        ONLINE_DENSE_MAPPING_MAX_TOTAL_POINTS > 0 &&
        static_cast<int>(points_.size()) >= ONLINE_DENSE_MAPPING_MAX_TOTAL_POINTS)
    {
        publish(header);
        return;
    }

    cv::Mat left_rect, right_rect, left_color_rect;
    cv::remap(left, left_rect, map00_, map01_, cv::INTER_LINEAR);
    cv::remap(right, right_rect, map10_, map11_, cv::INTER_LINEAR);
    if (!left_color.empty() && left_color.channels() == 3)
        cv::remap(left_color, left_color_rect, map00_, map01_, cv::INTER_LINEAR);

    cv::Mat disparity;
    if (!inferFoundationStereo(left_rect, right_rect, disparity))
    {
        if (++skipped_frames % 20 == 1)
            ROS_WARN("online dense skip: FoundationStereo inference failed");
        return;
    }

    cv::Mat grad_x, grad_y, grad;
    cv::Sobel(left_rect, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(left_rect, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad);

    std::vector<cv::Point> candidates;
    int sampled_pixels = 0;
    double depth_sum = 0.0;
    double depth_sq_sum = 0.0;
    candidates.reserve(static_cast<size_t>(ROW * COL / std::max(1, ONLINE_DENSE_MAPPING_PIXEL_STEP * ONLINE_DENSE_MAPPING_PIXEL_STEP)));
    for (int y = 0; y < ROW; y += std::max(1, ONLINE_DENSE_MAPPING_PIXEL_STEP))
    {
        for (int x = 0; x < COL; x += std::max(1, ONLINE_DENSE_MAPPING_PIXEL_STEP))
        {
            ++sampled_pixels;
            float d = disparity.at<float>(y, x);
            if (d < ONLINE_DENSE_MAPPING_MIN_DISPARITY || d > ONLINE_DENSE_MAPPING_MAX_DISPARITY)
                continue;
            if (grad.at<float>(y, x) < ONLINE_DENSE_MAPPING_MIN_GRADIENT)
                continue;
            double depth = fx_ * baseline_ / std::max(static_cast<double>(d), 1e-6);
            if (depth < ONLINE_DENSE_MAPPING_MIN_DEPTH || depth > ONLINE_DENSE_MAPPING_MAX_DEPTH)
                continue;
            depth_sum += depth;
            depth_sq_sum += depth * depth;
            candidates.emplace_back(x, y);
        }
    }

    if (candidates.empty())
    {
        ROS_WARN("online dense keyframe cloud: no valid depth candidates after filters");
        publish(header);
        return;
    }

    if (candidates.size() > static_cast<size_t>(ONLINE_DENSE_MAPPING_MAX_POINTS_PER_KEYFRAME))
    {
        std::vector<cv::Point> sampled;
        sampled.reserve(ONLINE_DENSE_MAPPING_MAX_POINTS_PER_KEYFRAME);
        for (int i = 0; i < ONLINE_DENSE_MAPPING_MAX_POINTS_PER_KEYFRAME; ++i)
        {
            size_t idx = static_cast<size_t>(i) * candidates.size() / ONLINE_DENSE_MAPPING_MAX_POINTS_PER_KEYFRAME;
            sampled.push_back(candidates[idx]);
        }
        candidates.swap(sampled);
    }

    const Eigen::Matrix3d r_bc = RIC[0];
    const Eigen::Vector3d t_bc = TIC[0];
    int added = 0;
    DenseSubmap submap;
    if (ONLINE_DENSE_MAPPING_USE_SUBMAP_MANAGER)
    {
        submap.stamp_key = submap_stamp_key;
        submap.p_wb = p_wb;
        submap.r_wb = r_wb;
        submap.local_points.reserve(candidates.size());
        submap.colors.reserve(candidates.size());
    }
    for (const auto &uv : candidates)
    {
        float d = disparity.at<float>(uv.y, uv.x);
        double z = fx_ * baseline_ / std::max(static_cast<double>(d), 1e-6);
        Eigen::Vector3d p_cam((uv.x - cx_) * z / fx_, (uv.y - cy_) * z / fx_, z);
        Eigen::Vector3d p_body = r_bc * p_cam + t_bc;
        uint32_t color = !left_color_rect.empty() ? bgrToRgb(left_color_rect.at<cv::Vec3b>(uv.y, uv.x))
                                                  : grayToRgb(left_rect.at<uint8_t>(uv.y, uv.x));
        if (ONLINE_DENSE_MAPPING_USE_SUBMAP_MANAGER)
        {
            submap.local_points.push_back(p_body.cast<float>());
            submap.colors.push_back(color);
            ++added;
            continue;
        }

        Eigen::Vector3d p_world_d = r_wb * p_body + p_wb;
        Eigen::Vector3f p_world = p_world_d.cast<float>();
        int64_t key = voxelKey(p_world);
        if (occupied_voxels_.find(key) != occupied_voxels_.end())
            continue;
        occupied_voxels_.insert(key);
        points_.push_back(p_world);
        colors_.push_back(color);
        ++added;
        if (ONLINE_DENSE_MAPPING_MAX_TOTAL_POINTS > 0 &&
            static_cast<int>(points_.size()) >= ONLINE_DENSE_MAPPING_MAX_TOTAL_POINTS)
            break;
    }

    if (ONLINE_DENSE_MAPPING_USE_SUBMAP_MANAGER)
    {
        const size_t raw_points = submap.local_points.size();
        voxelFilterLocalCloud(submap.local_points, submap.colors, ONLINE_DENSE_MAPPING_VOXEL_SIZE);
        if (ONLINE_DENSE_MAPPING_USE_LOCAL_RADIUS_FILTER)
        {
            radiusFilterLocalCloud(submap.local_points,
                                   submap.colors,
                                   ONLINE_DENSE_MAPPING_RADIUS_FILTER_RADIUS,
                                   ONLINE_DENSE_MAPPING_RADIUS_FILTER_MIN_NEIGHBORS);
        }

        const double valid_ratio = sampled_pixels > 0 ? static_cast<double>(candidates.size()) / sampled_pixels : 0.0;
        const double mean_depth = candidates.empty() ? 0.0 : depth_sum / candidates.size();
        const double depth_var = candidates.empty() ? 0.0 : std::max(0.0, depth_sq_sum / candidates.size() - mean_depth * mean_depth);
        const double depth_std = std::sqrt(depth_var);
        bool keep_submap = !submap.local_points.empty();
        if (ONLINE_DENSE_MAPPING_USE_SUBMAP_QUALITY_FILTER)
        {
            keep_submap = static_cast<int>(submap.local_points.size()) >= ONLINE_DENSE_MAPPING_MIN_SUBMAP_POINTS &&
                          valid_ratio >= ONLINE_DENSE_MAPPING_MIN_VALID_DEPTH_RATIO &&
                          depth_std <= ONLINE_DENSE_MAPPING_MAX_DEPTH_STD;
        }

        if (!keep_submap)
        {
            ROS_WARN("online dense submap rejected: raw %zu filtered %zu valid_ratio %.4f depth_std %.3f",
                     raw_points, submap.local_points.size(), valid_ratio, depth_std);
            publishSubmaps(header);
            return;
        }

        if (!submap.local_points.empty())
        {
            submaps_.push_back(std::move(submap));
            if (ONLINE_DENSE_MAPPING_USE_POSE_UPDATE_REASSEMBLY)
                submap_index_by_stamp_[submap_stamp_key] = submaps_.size() - 1;
        }
        ROS_INFO("online dense submap: candidates %zu, raw %zu, filtered %zu, valid_ratio %.4f, depth_std %.3f, submaps %zu",
                 candidates.size(), raw_points, submaps_.back().local_points.size(), valid_ratio, depth_std, submaps_.size());
        publishSubmaps(header);
    }
    else
    {
        ROS_INFO("online dense keyframe cloud: candidates %zu, added %d, total %zu", candidates.size(), added, points_.size());
        publish(header);
    }
}

void OnlineDenseMapper::publishSubmaps(const std_msgs::msg::Header &header)
{
    if (!pub_cloud_ || submaps_.empty())
        return;

    struct VoxelAccum
    {
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        uint64_t r = 0;
        uint64_t g = 0;
        uint64_t b = 0;
        uint32_t count = 0;
    };
    std::unordered_map<int64_t, VoxelAccum> voxels;
    const Eigen::Vector3d current_position = submaps_.back().p_wb;
    const size_t first_recent =
        ONLINE_DENSE_MAPPING_USE_ACTIVE_SUBMAP_WINDOW &&
                submaps_.size() > static_cast<size_t>(ONLINE_DENSE_MAPPING_MAX_ACTIVE_SUBMAPS)
            ? submaps_.size() - static_cast<size_t>(ONLINE_DENSE_MAPPING_MAX_ACTIVE_SUBMAPS)
            : 0;

    std::vector<size_t> active_indices;
    active_indices.reserve(submaps_.size() - first_recent);
    for (size_t i = first_recent; i < submaps_.size(); ++i)
    {
        if (ONLINE_DENSE_MAPPING_USE_ACTIVE_SUBMAP_WINDOW &&
            ONLINE_DENSE_MAPPING_ACTIVE_RADIUS > 0.0 &&
            (submaps_[i].p_wb - current_position).norm() > ONLINE_DENSE_MAPPING_ACTIVE_RADIUS)
        {
            continue;
        }
        active_indices.push_back(i);
    }
    if (active_indices.empty())
        active_indices.push_back(submaps_.size() - 1);

    size_t reserve_size = 0;
    for (size_t idx : active_indices)
    {
        const auto &submap = submaps_[idx];
        reserve_size += submap.local_points.size();
    }
    if (ONLINE_DENSE_MAPPING_MAX_TOTAL_POINTS > 0)
        reserve_size = std::min(reserve_size, static_cast<size_t>(ONLINE_DENSE_MAPPING_MAX_TOTAL_POINTS) * 2);
    voxels.reserve(reserve_size);

    for (size_t idx : active_indices)
    {
        const auto &submap = submaps_[idx];
        for (size_t i = 0; i < submap.local_points.size(); ++i)
        {
            Eigen::Vector3d p_body = submap.local_points[i].cast<double>();
            Eigen::Vector3f p_world = (submap.r_wb * p_body + submap.p_wb).cast<float>();
            int64_t key = voxelKey(p_world);
            VoxelAccum &acc = voxels[key];
            acc.sum += p_world;
            uint32_t rgb = submap.colors[i];
            acc.r += (rgb >> 16) & 0xff;
            acc.g += (rgb >> 8) & 0xff;
            acc.b += rgb & 0xff;
            ++acc.count;
        }
    }

    std::vector<Eigen::Vector3f> assembled_points;
    std::vector<uint32_t> assembled_colors;
    assembled_points.reserve(voxels.size());
    assembled_colors.reserve(voxels.size());
    const double hit_log_odds = logOdds(ONLINE_DENSE_MAPPING_OCCUPANCY_PROB_HIT);
    const double occupancy_threshold_log_odds = logOdds(ONLINE_DENSE_MAPPING_OCCUPANCY_THRESHOLD);
    int occupied_voxels = 0;
    for (const auto &kv : voxels)
    {
        const VoxelAccum &acc = kv.second;
        if (acc.count == 0)
            continue;
        if (ONLINE_DENSE_MAPPING_USE_OCCUPANCY_FUSION)
        {
            const double occupancy_log_odds = static_cast<double>(acc.count) * hit_log_odds;
            if (static_cast<int>(acc.count) < ONLINE_DENSE_MAPPING_OCCUPANCY_MIN_HITS ||
                occupancy_log_odds < occupancy_threshold_log_odds)
                continue;
        }
        ++occupied_voxels;
        assembled_points.push_back(acc.sum / static_cast<float>(acc.count));
        uint32_t r = static_cast<uint32_t>(acc.r / acc.count);
        uint32_t g = static_cast<uint32_t>(acc.g / acc.count);
        uint32_t b = static_cast<uint32_t>(acc.b / acc.count);
        assembled_colors.push_back((r << 16) | (g << 8) | b);
        if (ONLINE_DENSE_MAPPING_MAX_TOTAL_POINTS > 0 &&
            static_cast<int>(assembled_points.size()) >= ONLINE_DENSE_MAPPING_MAX_TOTAL_POINTS)
            break;
    }

    std::vector<Eigen::Vector3f> old_points;
    std::vector<uint32_t> old_colors;
    old_points.swap(points_);
    old_colors.swap(colors_);
    points_.swap(assembled_points);
    colors_.swap(assembled_colors);
    publish(header);
    points_.swap(old_points);
    colors_.swap(old_colors);
    if (ONLINE_DENSE_MAPPING_USE_ACTIVE_SUBMAP_WINDOW)
    {
        ROS_INFO("online dense active submaps: %zu / %zu", active_indices.size(), submaps_.size());
    }
    if (ONLINE_DENSE_MAPPING_USE_OCCUPANCY_FUSION)
    {
        ROS_INFO("online dense occupancy voxels: %d / %zu", occupied_voxels, voxels.size());
    }
}

void OnlineDenseMapper::publish(const std_msgs::msg::Header &header)
{
    if (!pub_cloud_ || points_.empty())
        return;

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header = header;
    cloud.header.frame_id = WORLD_FRAME_ID;
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(points_.size());
    cloud.is_bigendian = false;
    cloud.is_dense = false;
    cloud.point_step = 16;
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.fields.resize(4);
    cloud.fields[0].name = "x";
    cloud.fields[0].offset = 0;
    cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[0].count = 1;
    cloud.fields[1].name = "y";
    cloud.fields[1].offset = 4;
    cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[1].count = 1;
    cloud.fields[2].name = "z";
    cloud.fields[2].offset = 8;
    cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[2].count = 1;
    cloud.fields[3].name = "rgb";
    cloud.fields[3].offset = 12;
    cloud.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[3].count = 1;
    cloud.data.resize(static_cast<size_t>(cloud.row_step));
    for (size_t i = 0; i < points_.size(); ++i)
    {
        uint8_t *ptr = cloud.data.data() + i * cloud.point_step;
        float x = points_[i].x();
        float y = points_[i].y();
        float z = points_[i].z();
        uint32_t rgb = colors_[i];
        std::memcpy(ptr + 0, &x, sizeof(float));
        std::memcpy(ptr + 4, &y, sizeof(float));
        std::memcpy(ptr + 8, &z, sizeof(float));
        std::memcpy(ptr + 12, &rgb, sizeof(uint32_t));
    }
    pub_cloud_->publish(cloud);
}

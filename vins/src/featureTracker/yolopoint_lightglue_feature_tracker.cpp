/*******************************************************
 * YOLOPointv11 + LightGlue frontend for VINS-Fusion ROS2.
 *******************************************************/

#include "yolopoint_lightglue_feature_tracker.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

using std::map;
using std::pair;
using std::set;
using std::string;
using std::vector;

namespace
{
constexpr int kDescriptorDimension = 128;
constexpr int kCellSize = 8;

const vector<string> &cocoClassNames()
{
    static const vector<string> names{
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"};
    return names;
}

bool isPotentiallyDynamicClass(int class_id)
{
    // DynaSLAM-style prior: discard features on people, vehicles and animals.
    return (class_id >= 0 && class_id <= 8) ||
           (class_id >= 14 && class_id <= 23);
}

cv::Mat toGray8(const cv::Mat &image)
{
    if (image.empty())
        return cv::Mat();
    if (image.type() == CV_8UC1)
        return image;

    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else
        image.convertTo(gray, CV_8UC1);
    return gray;
}

cv::Mat toBgr(const cv::Mat &image)
{
    if (image.channels() == 3)
        return image.clone();
    cv::Mat bgr;
    cv::cvtColor(toGray8(image), bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}
}  // namespace

YOLOPointLightGlueFeatureTracker::YOLOPointLightGlueFeatureTracker() = default;

void YOLOPointLightGlueFeatureTracker::shutdown()
{
    initialized_ = false;
    cuda_postprocessor_.reset();
    matcher_session_.reset();
    extractor_session_.reset();
}

std::unique_ptr<Ort::Session> YOLOPointLightGlueFeatureTracker::createSession(
    const string &model_path,
    const string &cache_path,
    const std::unordered_map<string, string> &trt_profile)
{
    auto appendCuda = [](Ort::SessionOptions &options) {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        cuda_options.gpu_mem_limit = 0;
        cuda_options.arena_extend_strategy = 1;
        cuda_options.do_copy_in_default_stream = 1;
        options.AppendExecutionProvider_CUDA(cuda_options);
    };

    auto makeOptions = [&](bool with_tensorrt) {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(1);
        if (with_tensorrt)
        {
            vector<string> key_storage{
                "device_id",
                "trt_fp16_enable",
                "trt_engine_cache_enable",
                "trt_engine_cache_path",
                "trt_max_workspace_size",
                "trt_builder_optimization_level",
                "trt_min_subgraph_size"};
            vector<string> value_storage{
                "0", "1", "1", cache_path,
                std::to_string(4ULL << 30), "3", "1"};
            for (const auto &entry : trt_profile)
            {
                key_storage.push_back(entry.first);
                value_storage.push_back(entry.second);
            }
            vector<const char *> keys;
            vector<const char *> values;
            for (const string &key : key_storage)
                keys.push_back(key.c_str());
            for (const string &value : value_storage)
                values.push_back(value.c_str());

            const OrtApi &api = Ort::GetApi();
            OrtTensorRTProviderOptionsV2 *trt_options = nullptr;
            Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt_options));
            try
            {
                Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
                    trt_options, keys.data(), values.data(), keys.size()));
                options.AppendExecutionProvider_TensorRT_V2(*trt_options);
            }
            catch (...)
            {
                api.ReleaseTensorRTProviderOptions(trt_options);
                throw;
            }
            api.ReleaseTensorRTProviderOptions(trt_options);
        }
        if (YOLOPOINT_LIGHTGLUE_USE_CUDA)
            appendCuda(options);
        return options;
    };

    if (YOLOPOINT_LIGHTGLUE_USE_TENSORRT)
    {
        try
        {
            Ort::SessionOptions options = makeOptions(true);
            return std::unique_ptr<Ort::Session>(
                new Ort::Session(ort_env_, model_path.c_str(), options));
        }
        catch (const std::exception &error)
        {
            if (!YOLOPOINT_LIGHTGLUE_USE_CUDA)
                throw;
            ROS_WARN("YOLOPoint+LightGlue TensorRT session failed for %s: %s",
                     model_path.c_str(), error.what());
            ROS_WARN("YOLOPoint+LightGlue falling back to CUDAExecutionProvider");
        }
    }

    Ort::SessionOptions options = makeOptions(false);
    return std::unique_ptr<Ort::Session>(
        new Ort::Session(ort_env_, model_path.c_str(), options));
}

void YOLOPointLightGlueFeatureTracker::initializeModels()
{
    if (initialized_)
        return;

    const string extractor_shape =
        "image:1x3x" + std::to_string(YOLOPOINT_LIGHTGLUE_INPUT_HEIGHT) + "x" +
        std::to_string(YOLOPOINT_LIGHTGLUE_INPUT_WIDTH);
    const string extractor_min_shape = YOLOPOINT_OBJECT_DETECTION_ENABLE
        ? "image:1x3x384x768" : extractor_shape;
    const string extractor_opt_shape = YOLOPOINT_OBJECT_DETECTION_ENABLE
        ? "image:1x3x480x768" : extractor_shape;
    const string extractor_max_shape = YOLOPOINT_OBJECT_DETECTION_ENABLE
        ? "image:1x3x480x1248" : extractor_shape;
    std::unordered_map<string, string> extractor_profile{
        {"trt_profile_min_shapes", extractor_min_shape},
        {"trt_profile_opt_shapes", extractor_opt_shape},
        {"trt_profile_max_shapes", extractor_max_shape}};

    const string size_shape = "image_size0:1x2,image_size1:1x2";
    std::unordered_map<string, string> matcher_profile{
        {"trt_profile_min_shapes",
         "keypoints0:1x1x2,keypoints1:1x1x2,"
         "descriptors0:1x1x128,descriptors1:1x1x128," +
             size_shape},
        {"trt_profile_opt_shapes",
         "keypoints0:1x256x2,keypoints1:1x256x2,"
         "descriptors0:1x256x128,descriptors1:1x256x128," +
             size_shape},
        {"trt_profile_max_shapes",
         "keypoints0:1x512x2,keypoints1:1x512x2,"
         "descriptors0:1x512x128,descriptors1:1x512x128," +
             size_shape}};

    const string &extractor_model_path = YOLOPOINT_OBJECT_DETECTION_ENABLE
        ? YOLOPOINT_FULL_MODEL_PATH
        : YOLOPOINT_LIGHTGLUE_EXTRACTOR_MODEL_PATH;
    const string &extractor_cache_path = YOLOPOINT_OBJECT_DETECTION_ENABLE
        ? YOLOPOINT_FULL_TRT_CACHE_PATH
        : YOLOPOINT_LIGHTGLUE_EXTRACTOR_TRT_CACHE_PATH;
    extractor_session_ = createSession(
        extractor_model_path,
        extractor_cache_path,
        extractor_profile);
    object_detection_available_ = false;
    for (size_t index = 0; index < extractor_session_->GetOutputCount(); ++index)
    {
        auto name = extractor_session_->GetOutputNameAllocated(index, ort_allocator_);
        if (name && string(name.get()) == "detections")
            object_detection_available_ = true;
    }
    if (YOLOPOINT_OBJECT_DETECTION_ENABLE && !object_detection_available_)
        throw std::runtime_error(
            "YOLOPoint object detection is enabled, but the extractor has no detections output");
    matcher_session_ = createSession(
        YOLOPOINT_LIGHTGLUE_MATCHER_MODEL_PATH,
        YOLOPOINT_LIGHTGLUE_MATCHER_TRT_CACHE_PATH,
        matcher_profile);
    if (YOLOPOINT_LIGHTGLUE_USE_CUDA)
    {
        cuda_postprocessor_.reset(new YOLOPointCudaPostprocessor(
            YOLOPOINT_LIGHTGLUE_INPUT_WIDTH,
            YOLOPOINT_LIGHTGLUE_INPUT_HEIGHT,
            kDescriptorDimension,
            YOLOPOINT_LIGHTGLUE_MAX_KEYPOINTS));
    }
    initialized_ = true;
    ROS_INFO("YOLOPoint+LightGlue models initialized");
    ROS_INFO("  extractor: %s", extractor_model_path.c_str());
    ROS_INFO("  matcher:   %s", YOLOPOINT_LIGHTGLUE_MATCHER_MODEL_PATH.c_str());
    ROS_INFO("  extractor postprocess: %s",
             cuda_postprocessor_ && !YOLOPOINT_OBJECT_DETECTION_ENABLE
                 ? "CUDA" : "CPU");
    ROS_INFO("  object detections: %s",
             object_detection_available_ ? "available" : "not present");
}

bool YOLOPointLightGlueFeatureTracker::inBorder(
    const cv::Point2f &point, int width, int height) const
{
    const int x = cvRound(point.x);
    const int y = cvRound(point.y);
    return x >= 1 && y >= 1 && x < width - 1 && y < height - 1;
}

void YOLOPointLightGlueFeatureTracker::decodeExtractor(
    const float *semi,
    const vector<int64_t> &semi_shape,
    const float *dense_descriptors,
    const vector<int64_t> &descriptor_shape,
    float scale_x,
    float scale_y,
    ExtractedFeatures &features) const
{
    if (semi_shape.size() != 4 || descriptor_shape.size() != 4 ||
        semi_shape[0] != 1 || semi_shape[1] != 65 ||
        descriptor_shape[0] != 1 ||
        descriptor_shape[1] != kDescriptorDimension)
        throw std::runtime_error("unexpected YOLOPoint extractor output shape");

    const int coarse_h = static_cast<int>(semi_shape[2]);
    const int coarse_w = static_cast<int>(semi_shape[3]);
    const int image_h = coarse_h * kCellSize;
    const int image_w = coarse_w * kCellSize;
    const int coarse_area = coarse_h * coarse_w;
    cv::Mat score_map(image_h, image_w, CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < coarse_h; ++y)
    {
        for (int x = 0; x < coarse_w; ++x)
        {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (int channel = 0; channel < 65; ++channel)
                max_logit = std::max(
                    max_logit, semi[channel * coarse_area + y * coarse_w + x]);

            float sum = 0.0f;
            float probabilities[64];
            for (int channel = 0; channel < 65; ++channel)
            {
                const float value = std::exp(
                    semi[channel * coarse_area + y * coarse_w + x] - max_logit);
                sum += value;
                if (channel < 64)
                    probabilities[channel] = value;
            }
            for (int channel = 0; channel < 64; ++channel)
            {
                const int dy = channel / kCellSize;
                const int dx = channel % kCellSize;
                score_map.at<float>(y * kCellSize + dy, x * kCellSize + dx) =
                    probabilities[channel] / sum;
            }
        }
    }
    cv::Mat maxima;
    const int radius = YOLOPOINT_LIGHTGLUE_NMS_RADIUS;
    if (radius > 0)
    {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(2 * radius + 1, 2 * radius + 1));
        cv::Mat pooled;
        cv::dilate(score_map, pooled, kernel);
        cv::compare(score_map, pooled, maxima, cv::CMP_EQ);
        for (int iteration = 0; iteration < 2; ++iteration)
        {
            cv::Mat suppressed, suppression_mask, remaining_scores, new_pool, new_maxima;
            cv::dilate(maxima, suppressed, kernel);
            cv::compare(suppressed, 0, suppression_mask, cv::CMP_GT);
            remaining_scores = score_map.clone();
            remaining_scores.setTo(0, suppression_mask);
            cv::dilate(remaining_scores, new_pool, kernel);
            cv::compare(remaining_scores, new_pool, new_maxima, cv::CMP_EQ);
            cv::bitwise_not(suppression_mask, suppression_mask);
            cv::bitwise_and(new_maxima, suppression_mask, new_maxima);
            cv::bitwise_or(maxima, new_maxima, maxima);
        }
    }
    else
    {
        maxima = cv::Mat(score_map.size(), CV_8UC1, cv::Scalar(255));
    }
    vector<cv::KeyPoint> candidates;
    const int border = YOLOPOINT_LIGHTGLUE_REMOVE_BORDERS;
    for (int y = border; y < image_h - border; ++y)
    {
        const float *score_row = score_map.ptr<float>(y);
        const unsigned char *maxima_row = maxima.ptr<unsigned char>(y);
        for (int x = border; x < image_w - border; ++x)
        {
            if (!maxima_row[x] ||
                score_row[x] <= static_cast<float>(YOLOPOINT_LIGHTGLUE_SCORE_THRESHOLD))
                continue;
            cv::KeyPoint keypoint;
            keypoint.pt = cv::Point2f((x + 0.5f) * scale_x,
                                      (y + 0.5f) * scale_y);
            keypoint.response = score_row[x];
            keypoint.size = 1.0f;
            keypoint.class_id = y * image_w + x;
            candidates.push_back(keypoint);
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
                  return a.response > b.response;
              });
    if (static_cast<int>(candidates.size()) > YOLOPOINT_LIGHTGLUE_MAX_KEYPOINTS)
        candidates.resize(YOLOPOINT_LIGHTGLUE_MAX_KEYPOINTS);
    const int descriptor_h = static_cast<int>(descriptor_shape[2]);
    const int descriptor_w = static_cast<int>(descriptor_shape[3]);
    const int descriptor_area = descriptor_h * descriptor_w;
    features.keypoints = candidates;
    features.descriptors = cv::Mat(
        static_cast<int>(candidates.size()), kDescriptorDimension, CV_32FC1);

    for (int index = 0; index < static_cast<int>(candidates.size()); ++index)
    {
        const int packed = candidates[index].class_id;
        const int pixel_y = packed / image_w;
        const int pixel_x = packed % image_w;
        const float descriptor_x = (pixel_x + 0.5f) / kCellSize - 0.5f;
        const float descriptor_y = (pixel_y + 0.5f) / kCellSize - 0.5f;
        const int x0 = static_cast<int>(std::floor(descriptor_x));
        const int y0 = static_cast<int>(std::floor(descriptor_y));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const float wx = descriptor_x - x0;
        const float wy = descriptor_y - y0;

        float norm_squared = 0.0f;
        float *output = features.descriptors.ptr<float>(index);
        for (int channel = 0; channel < kDescriptorDimension; ++channel)
        {
            auto sample = [&](int sx, int sy) {
                if (sx < 0 || sy < 0 || sx >= descriptor_w || sy >= descriptor_h)
                    return 0.0f;
                return dense_descriptors[
                    channel * descriptor_area + sy * descriptor_w + sx];
            };
            const float top = (1.0f - wx) * sample(x0, y0) + wx * sample(x1, y0);
            const float bottom =
                (1.0f - wx) * sample(x0, y1) + wx * sample(x1, y1);
            output[channel] = (1.0f - wy) * top + wy * bottom;
            norm_squared += output[channel] * output[channel];
        }
        const float inverse_norm =
            1.0f / std::sqrt(std::max(norm_squared, 1e-12f));
        for (int channel = 0; channel < kDescriptorDimension; ++channel)
            output[channel] *= inverse_norm;
    }
}

void YOLOPointLightGlueFeatureTracker::decodeDetections(
    const float *predictions,
    const vector<int64_t> &shape,
    float scale_x,
    float scale_y,
    vector<Detection> &detections) const
{
    detections.clear();
    if (shape.size() != 3 || shape[0] != 1 || shape[2] < 6)
        throw std::runtime_error("unexpected YOLOPoint detections output shape");

    const int prediction_count = static_cast<int>(shape[1]);
    const int values_per_prediction = static_cast<int>(shape[2]);
    const int class_count = values_per_prediction - 5;
    vector<Detection> candidates;
    candidates.reserve(std::min(prediction_count, 1024));
    for (int index = 0; index < prediction_count; ++index)
    {
        const float *prediction = predictions + index * values_per_prediction;
        const float object_confidence = prediction[4];
        if (!std::isfinite(object_confidence) ||
            object_confidence < YOLOPOINT_OBJECT_CONFIDENCE_THRESHOLD)
            continue;

        int best_class = 0;
        float best_class_confidence = prediction[5];
        for (int class_id = 1; class_id < class_count; ++class_id)
        {
            if (prediction[5 + class_id] > best_class_confidence)
            {
                best_class_confidence = prediction[5 + class_id];
                best_class = class_id;
            }
        }
        const float confidence = object_confidence * best_class_confidence;
        if (!std::isfinite(confidence) ||
            confidence < YOLOPOINT_OBJECT_CONFIDENCE_THRESHOLD)
            continue;

        const float width = prediction[2] * scale_x;
        const float height = prediction[3] * scale_y;
        const float left = (prediction[0] - 0.5f * prediction[2]) * scale_x;
        const float top = (prediction[1] - 0.5f * prediction[3]) * scale_y;
        if (!std::isfinite(left) || !std::isfinite(top) ||
            !std::isfinite(width) || !std::isfinite(height) ||
            width <= 1.0f || height <= 1.0f)
            continue;

        Detection detection;
        detection.box = cv::Rect2f(left, top, width, height);
        detection.confidence = confidence;
        detection.class_id = best_class;
        candidates.push_back(detection);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Detection &left, const Detection &right) {
                  return left.confidence > right.confidence;
              });
    auto intersectionOverUnion = [](const cv::Rect2f &left,
                                    const cv::Rect2f &right) {
        const float intersection = (left & right).area();
        const float union_area = left.area() + right.area() - intersection;
        return union_area > 0.0f ? intersection / union_area : 0.0f;
    };
    for (const Detection &candidate : candidates)
    {
        bool suppressed = false;
        for (const Detection &kept : detections)
        {
            if (candidate.class_id == kept.class_id &&
                intersectionOverUnion(candidate.box, kept.box) >
                    YOLOPOINT_OBJECT_IOU_THRESHOLD)
            {
                suppressed = true;
                break;
            }
        }
        if (!suppressed)
        {
            detections.push_back(candidate);
            if (static_cast<int>(detections.size()) >=
                YOLOPOINT_OBJECT_MAX_DETECTIONS)
                break;
        }
    }
}

YOLOPointLightGlueFeatureTracker::ExtractedFeatures
YOLOPointLightGlueFeatureTracker::extractLeft(const cv::Mat &image)
{
    if (!initialized_)
        throw std::runtime_error("YOLOPoint+LightGlue models are not initialized");

    const int64 preprocess_start_ticks = cv::getTickCount();
    const cv::Mat gray = toGray8(image);
    cv::Mat resized;
    if (gray.cols != YOLOPOINT_LIGHTGLUE_INPUT_WIDTH ||
        gray.rows != YOLOPOINT_LIGHTGLUE_INPUT_HEIGHT)
        cv::resize(gray, resized,
                   cv::Size(YOLOPOINT_LIGHTGLUE_INPUT_WIDTH,
                            YOLOPOINT_LIGHTGLUE_INPUT_HEIGHT));
    else
        resized = gray;

    const int input_h = resized.rows;
    const int input_w = resized.cols;
    const int image_area = input_h * input_w;
    vector<float> input_tensor(3 * image_area);
    for (int y = 0; y < input_h; ++y)
    {
        const unsigned char *row = resized.ptr<unsigned char>(y);
        for (int x = 0; x < input_w; ++x)
        {
            const float value = row[x] / 255.0f;
            const int offset = y * input_w + x;
            input_tensor[offset] = value;
            input_tensor[image_area + offset] = value;
            input_tensor[2 * image_area + offset] = value;
        }
    }

    const vector<int64_t> input_shape{1, 3, input_h, input_w};
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                   OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor.size(),
        input_shape.data(), input_shape.size());
    const char *input_names[] = {"image"};
    const char *output_names[] = {"semi", "descriptors"};
    const int64 inference_start_ticks = cv::getTickCount();
    ExtractedFeatures features;
    int64 decode_start_ticks = 0;
    if (cuda_postprocessor_ && !YOLOPOINT_OBJECT_DETECTION_ENABLE)
    {
        const vector<int64_t> semi_shape{
            1, 65, input_h / kCellSize, input_w / kCellSize};
        const vector<int64_t> descriptor_shape{
            1, kDescriptorDimension,
            input_h / kCellSize, input_w / kCellSize};
        Ort::MemoryInfo cuda_memory_info(
            "Cuda", OrtAllocatorType::OrtDeviceAllocator,
            0, OrtMemTypeDefault);
        Ort::Value semi_output = Ort::Value::CreateTensor<float>(
            cuda_memory_info, cuda_postprocessor_->semiDeviceData(),
            cuda_postprocessor_->semiElementCount(),
            semi_shape.data(), semi_shape.size());
        Ort::Value descriptor_output = Ort::Value::CreateTensor<float>(
            cuda_memory_info,
            cuda_postprocessor_->denseDescriptorDeviceData(),
            cuda_postprocessor_->denseDescriptorElementCount(),
            descriptor_shape.data(), descriptor_shape.size());
        Ort::IoBinding binding(*extractor_session_);
        binding.BindInput(input_names[0], input);
        binding.BindOutput(output_names[0], semi_output);
        binding.BindOutput(output_names[1], descriptor_output);
        extractor_session_->Run(Ort::RunOptions{nullptr}, binding);
        binding.SynchronizeOutputs();
        decode_start_ticks = cv::getTickCount();
        cuda_postprocessor_->process(
            static_cast<float>(YOLOPOINT_LIGHTGLUE_SCORE_THRESHOLD),
            YOLOPOINT_LIGHTGLUE_NMS_RADIUS,
            YOLOPOINT_LIGHTGLUE_REMOVE_BORDERS,
            static_cast<float>(image.cols) / input_w,
            static_cast<float>(image.rows) / input_h,
            features.keypoints,
            features.descriptors);
        if (LOOP_CLOSURE_ENABLE && LOOP_CLOSURE_MODE == 3)
            cuda_postprocessor_->globalDescriptor(features.global_descriptor);
    }
    else
    {
        vector<const char *> requested_outputs{"semi", "descriptors"};
        if (YOLOPOINT_OBJECT_DETECTION_ENABLE)
            requested_outputs.push_back("detections");
        auto outputs = extractor_session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input, 1,
            requested_outputs.data(), requested_outputs.size());
        decode_start_ticks = cv::getTickCount();
        const vector<int64_t> semi_shape =
            outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const vector<int64_t> descriptor_shape =
            outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        decodeExtractor(
            outputs[0].GetTensorData<float>(),
            semi_shape,
            outputs[1].GetTensorData<float>(),
            descriptor_shape,
            static_cast<float>(image.cols) / input_w,
            static_cast<float>(image.rows) / input_h,
            features);
        if (YOLOPOINT_OBJECT_DETECTION_ENABLE)
        {
            const vector<int64_t> detection_shape =
                outputs[2].GetTensorTypeAndShapeInfo().GetShape();
            decodeDetections(
                outputs[2].GetTensorData<float>(), detection_shape,
                static_cast<float>(image.cols) / input_w,
                static_cast<float>(image.rows) / input_h,
                features.detections);
        }
        if (LOOP_CLOSURE_ENABLE && LOOP_CLOSURE_MODE == 3)
        {
            const float *dense = outputs[1].GetTensorData<float>();
            const int area = static_cast<int>(descriptor_shape[2] * descriptor_shape[3]);
            features.global_descriptor = cv::Mat::zeros(1, kDescriptorDimension, CV_32FC1);
            for (int channel = 0; channel < kDescriptorDimension; ++channel)
            {
                const float *values = dense + channel * area;
                double sum = 0.0;
                for (int index = 0; index < area; ++index)
                    sum += values[index];
                features.global_descriptor.at<float>(0, channel) =
                    static_cast<float>(sum / std::max(1, area));
            }
            cv::normalize(features.global_descriptor, features.global_descriptor, 1.0, 0.0,
                          cv::NORM_L2);
        }
    }
    const int64 decode_end_ticks = cv::getTickCount();
    const double tick_ms = 1000.0 / cv::getTickFrequency();
    extractor_preprocess_ms_ =
        (inference_start_ticks - preprocess_start_ticks) * tick_ms;
    extractor_inference_ms_ =
        (decode_start_ticks - inference_start_ticks) * tick_ms;
    extractor_decode_ms_ =
        (decode_end_ticks - decode_start_ticks) * tick_ms;
    return features;
}

void YOLOPointLightGlueFeatureTracker::matchTemporal(
    const ExtractedFeatures &current,
    vector<int> &matched_ids,
    vector<int> &matched_track_counts,
    vector<unsigned char> &matched_status)
{
    const int current_count = static_cast<int>(current.keypoints.size());
    matched_ids.assign(current_count, -1);
    matched_track_counts.assign(current_count, 0);
    matched_status.assign(current_count, 0);
    if (previous_keypoints_.empty() || current.keypoints.empty())
    {
        matcher_prepare_ms_ = 0.0;
        matcher_inference_ms_ = 0.0;
        matcher_postprocess_ms_ = 0.0;
        matcher_raw_matches_ = 0;
        return;
    }

    const int64 prepare_start_ticks = cv::getTickCount();
    const int previous_count = static_cast<int>(previous_keypoints_.size());
    vector<float> keypoints0(previous_count * 2);
    vector<float> keypoints1(current_count * 2);
    for (int i = 0; i < previous_count; ++i)
    {
        keypoints0[2 * i] = previous_keypoints_[i].pt.x;
        keypoints0[2 * i + 1] = previous_keypoints_[i].pt.y;
    }
    for (int i = 0; i < current_count; ++i)
    {
        keypoints1[2 * i] = current.keypoints[i].pt.x;
        keypoints1[2 * i + 1] = current.keypoints[i].pt.y;
    }

    cv::Mat previous_contiguous = previous_descriptors_.isContinuous()
                                      ? previous_descriptors_
                                      : previous_descriptors_.clone();
    cv::Mat current_contiguous = current.descriptors.isContinuous()
                                     ? current.descriptors
                                     : current.descriptors.clone();
    vector<float> image_size0{
        static_cast<float>(col_), static_cast<float>(row_)};
    vector<float> image_size1 = image_size0;
    const vector<int64_t> keypoint0_shape{1, previous_count, 2};
    const vector<int64_t> keypoint1_shape{1, current_count, 2};
    const vector<int64_t> descriptor0_shape{
        1, previous_count, kDescriptorDimension};
    const vector<int64_t> descriptor1_shape{
        1, current_count, kDescriptorDimension};
    const vector<int64_t> image_size_shape{1, 2};
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                   OrtMemTypeDefault);
    vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, keypoints0.data(), keypoints0.size(),
        keypoint0_shape.data(), keypoint0_shape.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, keypoints1.data(), keypoints1.size(),
        keypoint1_shape.data(), keypoint1_shape.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, previous_contiguous.ptr<float>(),
        previous_count * kDescriptorDimension,
        descriptor0_shape.data(), descriptor0_shape.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, current_contiguous.ptr<float>(),
        current_count * kDescriptorDimension,
        descriptor1_shape.data(), descriptor1_shape.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, image_size0.data(), image_size0.size(),
        image_size_shape.data(), image_size_shape.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, image_size1.data(), image_size1.size(),
        image_size_shape.data(), image_size_shape.size()));

    const char *input_names[] = {
        "keypoints0", "keypoints1", "descriptors0",
        "descriptors1", "image_size0", "image_size1"};
    const char *output_names[] = {"matches0", "matching_scores0"};
    const int64 inference_start_ticks = cv::getTickCount();
    auto outputs = matcher_session_->Run(
        Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(),
        output_names, 2);
    const int64 postprocess_start_ticks = cv::getTickCount();
    const int64_t *matches = outputs[0].GetTensorData<int64_t>();

    struct Match
    {
        int previous_index;
        int current_index;
    };
    vector<Match> valid_matches;
    vector<cv::Point2f> previous_points;
    vector<cv::Point2f> current_points;
    const auto virtualPinholePoint =
        [this](const cv::Point2f &raw_point)
    {
        Eigen::Vector3d ray;
        cameras_[0]->liftProjective(
            Eigen::Vector2d(raw_point.x, raw_point.y), ray);
        return cv::Point2f(
            static_cast<float>(
                FOCAL_LENGTH * ray.x() / ray.z() + col_ / 2.0),
            static_cast<float>(
                FOCAL_LENGTH * ray.y() / ray.z() + row_ / 2.0));
    };
    for (int previous_index = 0; previous_index < previous_count;
         ++previous_index)
    {
        const int64_t current_index = matches[previous_index];
        if (current_index < 0 || current_index >= current_count)
            continue;
        valid_matches.push_back(
            {previous_index, static_cast<int>(current_index)});
        // A single fundamental matrix is valid on the undistorted virtual
        // pinhole plane, not directly on raw MEI image coordinates.
        previous_points.push_back(
            virtualPinholePoint(previous_keypoints_[previous_index].pt));
        current_points.push_back(
            virtualPinholePoint(current.keypoints[current_index].pt));
    }

    vector<unsigned char> geometric_status(valid_matches.size(), 1);
    matcher_raw_matches_ = static_cast<int>(valid_matches.size());
    if (valid_matches.size() >= 8)
    {
        vector<unsigned char> ransac_status;
        cv::setRNGSeed(0x5eed);
        cv::Mat fundamental = cv::findFundamentalMat(
            previous_points, current_points, cv::FM_RANSAC,
            F_THRESHOLD, 0.99, ransac_status);
        if (!fundamental.empty() && ransac_status.size() == valid_matches.size())
            geometric_status.swap(ransac_status);
    }

    for (int i = 0; i < static_cast<int>(valid_matches.size()); ++i)
    {
        if (!geometric_status[i])
            continue;
        const Match &match = valid_matches[i];
        matched_ids[match.current_index] = previous_ids_[match.previous_index];
        matched_track_counts[match.current_index] =
            previous_track_counts_[match.previous_index] + 1;
        matched_status[match.current_index] = 1;
    }
    const int64 postprocess_end_ticks = cv::getTickCount();
    const double tick_ms = 1000.0 / cv::getTickFrequency();
    matcher_prepare_ms_ =
        (inference_start_ticks - prepare_start_ticks) * tick_ms;
    matcher_inference_ms_ =
        (postprocess_start_ticks - inference_start_ticks) * tick_ms;
    matcher_postprocess_ms_ =
        (postprocess_end_ticks - postprocess_start_ticks) * tick_ms;
}

void YOLOPointLightGlueFeatureTracker::trackTemporalGeometry(
    const cv::Mat &current_left,
    map<int, cv::Point2f> &tracked_points,
    map<int, cv::Point2f> *raw_tracked_points)
{
    tracked_points.clear();
    if (raw_tracked_points)
        raw_tracked_points->clear();
    temporal_klt_fb_matches_ = 0;
    temporal_klt_f_matches_ = 0;
    const bool temporal_klt_needed =
        YOLOPOINT_LIGHTGLUE_USE_PERSISTENT_KLT_GEOMETRY;
    if (!temporal_klt_needed ||
        previous_left_image_.empty() || previous_ids_.empty() ||
        previous_left_pixel_map_.empty())
        return;

    vector<int> ids;
    vector<cv::Point2f> previous_points;
    ids.reserve(previous_ids_.size());
    previous_points.reserve(previous_ids_.size());
    for (int id : previous_ids_)
    {
        const auto point = previous_left_pixel_map_.find(id);
        if (point == previous_left_pixel_map_.end())
            continue;
        ids.push_back(id);
        previous_points.push_back(point->second);
    }
    if (previous_points.empty())
        return;

    const cv::Mat current_gray = toGray8(current_left);
    vector<cv::Point2f> current_points;
    vector<unsigned char> forward_status;
    vector<float> errors;
    cv::calcOpticalFlowPyrLK(
        previous_left_image_, current_gray,
        previous_points, current_points,
        forward_status, errors, cv::Size(21, 21), 3);

    vector<cv::Point2f> reverse_points;
    vector<unsigned char> reverse_status;
    if (FLOW_BACK)
    {
        cv::calcOpticalFlowPyrLK(
            current_gray, previous_left_image_,
            current_points, reverse_points,
            reverse_status, errors, cv::Size(21, 21), 3);
    }

    vector<int> fb_ids;
    vector<cv::Point2f> fb_previous_points;
    vector<cv::Point2f> fb_current_points;
    fb_ids.reserve(ids.size());
    fb_previous_points.reserve(ids.size());
    fb_current_points.reserve(ids.size());
    for (int i = 0; i < static_cast<int>(ids.size()); ++i)
    {
        if (!forward_status[i] ||
            !inBorder(current_points[i], current_gray.cols, current_gray.rows))
            continue;
        if (FLOW_BACK)
        {
            if (i >= static_cast<int>(reverse_status.size()) ||
                !reverse_status[i] ||
                cv::norm(previous_points[i] - reverse_points[i]) > 0.5)
                continue;
        }
        fb_ids.push_back(ids[i]);
        fb_previous_points.push_back(previous_points[i]);
        fb_current_points.push_back(current_points[i]);
    }
    temporal_klt_fb_matches_ = static_cast<int>(fb_ids.size());
    if (raw_tracked_points)
    {
        for (int i = 0; i < static_cast<int>(fb_ids.size()); ++i)
            (*raw_tracked_points)[fb_ids[i]] = fb_current_points[i];
    }

    vector<unsigned char> fundamental_status(fb_ids.size(), 1);
    if (YOLOPOINT_LIGHTGLUE_PERSISTENT_KLT_USE_FUNDAMENTAL_FILTER &&
        fb_ids.size() >= 8 && !cameras_.empty())
    {
        vector<cv::Point2f> previous_virtual_points;
        vector<cv::Point2f> current_virtual_points;
        previous_virtual_points.reserve(fb_ids.size());
        current_virtual_points.reserve(fb_ids.size());
        const auto virtualPinholePoint =
            [this](const cv::Point2f &raw_point)
        {
            Eigen::Vector3d ray;
            cameras_[0]->liftProjective(
                Eigen::Vector2d(raw_point.x, raw_point.y), ray);
            return cv::Point2f(
                static_cast<float>(
                    FOCAL_LENGTH * ray.x() / ray.z() + col_ / 2.0),
                static_cast<float>(
                    FOCAL_LENGTH * ray.y() / ray.z() + row_ / 2.0));
        };
        for (int i = 0; i < static_cast<int>(fb_ids.size()); ++i)
        {
            previous_virtual_points.push_back(
                virtualPinholePoint(fb_previous_points[i]));
            current_virtual_points.push_back(
                virtualPinholePoint(fb_current_points[i]));
        }

        vector<unsigned char> ransac_status;
        cv::setRNGSeed(0x4b4c54);
        const cv::Mat fundamental = cv::findFundamentalMat(
            previous_virtual_points, current_virtual_points,
            cv::FM_RANSAC, F_THRESHOLD, 0.99, ransac_status);
        if (!fundamental.empty() &&
            ransac_status.size() == fundamental_status.size())
            fundamental_status.swap(ransac_status);
    }

    for (int i = 0; i < static_cast<int>(fb_ids.size()); ++i)
    {
        if (!fundamental_status[i])
            continue;
        tracked_points[fb_ids[i]] = fb_current_points[i];
    }
    temporal_klt_f_matches_ = static_cast<int>(tracked_points.size());
}

vector<cv::Point2f> YOLOPointLightGlueFeatureTracker::undistort(
    const vector<cv::Point2f> &points,
    const camodocal::CameraPtr &camera) const
{
    vector<cv::Point2f> result;
    result.reserve(points.size());
    for (const cv::Point2f &point : points)
    {
        Eigen::Vector3d ray;
        camera->liftProjective(Eigen::Vector2d(point.x, point.y), ray);
        result.emplace_back(
            static_cast<float>(ray.x() / ray.z()),
            static_cast<float>(ray.y() / ray.z()));
    }
    return result;
}

cv::Point2f YOLOPointLightGlueFeatureTracker::velocityFor(
    int id,
    const cv::Point2f &undistorted,
    map<int, cv::Point2f> &current_map,
    const map<int, cv::Point2f> &previous_map) const
{
    current_map[id] = undistorted;
    if (previous_time_ < 0.0 || current_time_ <= previous_time_)
        return cv::Point2f(0, 0);
    auto previous = previous_map.find(id);
    if (previous == previous_map.end())
        return cv::Point2f(0, 0);
    const double dt = current_time_ - previous_time_;
    return cv::Point2f(
        static_cast<float>((undistorted.x - previous->second.x) / dt),
        static_cast<float>((undistorted.y - previous->second.y) / dt));
}

void YOLOPointLightGlueFeatureTracker::matchStereoKlt(
    const cv::Mat &left,
    const cv::Mat &right,
    const vector<Observation> &left_observations,
    map<int, Observation> &right_observations)
{
    const int64 pyramid_start_ticks = cv::getTickCount();
    right_observations.clear();
    current_right_undistorted_map_.clear();
    stereo_fb_matches_ = 0;
    stereo_corner_matches_ = 0;
    stereo_epipolar_matches_ = 0;
    stereo_positive_depth_matches_ = 0;
    if (left_observations.empty() || right.empty() || cameras_.size() < 2 ||
        RIC.size() < 2 || TIC.size() < 2)
    {
        stereo_pyramid_ms_ = 0.0;
        stereo_flow_ms_ = 0.0;
        stereo_filter_ms_ = 0.0;
        return;
    }

    vector<cv::Point2f> left_points;
    left_points.reserve(left_observations.size());
    for (const Observation &observation : left_observations)
        left_points.push_back(observation.keypoint.pt);

    const cv::Mat left_gray = toGray8(left);
    const cv::Mat right_gray = toGray8(right);
    cv::Mat left_corner_eigenvalues;
    if (YOLOPOINT_LIGHTGLUE_STEREO_MIN_CORNER_EIGENVALUE > 0.0)
        cv::cornerMinEigenVal(
            left_gray, left_corner_eigenvalues, 7, 3);
    const cv::Size lk_window(21, 21);
    const int64 flow_start_ticks = cv::getTickCount();
    vector<cv::Point2f> right_points;
    vector<unsigned char> forward_status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(
        left_gray, right_gray, left_points, right_points,
        forward_status, error,
        lk_window, 3);

    vector<cv::Point2f> reverse_points;
    vector<unsigned char> reverse_status;
    if (FLOW_BACK)
    {
        cv::calcOpticalFlowPyrLK(
            right_gray, left_gray, right_points, reverse_points,
            reverse_status, error,
            lk_window, 3);
    }
    const int64 filter_start_ticks = cv::getTickCount();

    const Eigen::Matrix3d rotation_10 = RIC[1].transpose() * RIC[0];
    const Eigen::Vector3d translation_10 =
        RIC[1].transpose() * (TIC[0] - TIC[1]);
    Eigen::Matrix3d translation_skew;
    translation_skew <<
        0.0, -translation_10.z(), translation_10.y(),
        translation_10.z(), 0.0, -translation_10.x(),
        -translation_10.y(), translation_10.x(), 0.0;
    const Eigen::Matrix3d essential = translation_skew * rotation_10;

    vector<int> kept_indices;
    vector<cv::Point2f> kept_right_points;
    for (int i = 0; i < static_cast<int>(left_points.size()); ++i)
    {
        if (!forward_status[i])
            continue;
        if (FLOW_BACK)
        {
            if (i >= static_cast<int>(reverse_status.size()) ||
                !reverse_status[i] ||
                !inBorder(right_points[i], right_gray.cols, right_gray.rows))
                continue;
            const float dx = left_points[i].x - reverse_points[i].x;
            const float dy = left_points[i].y - reverse_points[i].y;
            if (std::sqrt(dx * dx + dy * dy) > 0.5f)
                continue;
        }

        ++stereo_fb_matches_;
        if (!left_corner_eigenvalues.empty())
        {
            const cv::Point point(
                cvRound(left_points[i].x),
                cvRound(left_points[i].y));
            if (point.x < 0 || point.y < 0 ||
                point.x >= left_corner_eigenvalues.cols ||
                point.y >= left_corner_eigenvalues.rows ||
                left_corner_eigenvalues.at<float>(point) <
                    YOLOPOINT_LIGHTGLUE_STEREO_MIN_CORNER_EIGENVALUE)
                continue;
        }
        ++stereo_corner_matches_;
        Eigen::Vector3d left_ray;
        Eigen::Vector3d right_ray;
        cameras_[0]->liftProjective(
            Eigen::Vector2d(left_points[i].x, left_points[i].y), left_ray);
        cameras_[1]->liftProjective(
            Eigen::Vector2d(right_points[i].x, right_points[i].y), right_ray);
        if (std::abs(left_ray.z()) <= 1e-12 ||
            std::abs(right_ray.z()) <= 1e-12)
            continue;
        left_ray /= left_ray.z();
        right_ray /= right_ray.z();

        const Eigen::Vector3d epipolar_line = essential * left_ray;
        const double epipolar_denominator = std::hypot(
            epipolar_line.x(), epipolar_line.y());
        if (epipolar_denominator <= 1e-12)
            continue;
        const double epipolar_error =
            std::abs(right_ray.dot(epipolar_line)) /
            epipolar_denominator * FOCAL_LENGTH;
        if (!std::isfinite(epipolar_error) ||
            epipolar_error >
                YOLOPOINT_LIGHTGLUE_STEREO_EPIPOLAR_THRESHOLD)
            continue;
        ++stereo_epipolar_matches_;

        Eigen::Matrix<double, 3, 2> ray_system;
        ray_system.col(0) = rotation_10 * left_ray;
        ray_system.col(1) = -right_ray;
        const Eigen::Vector2d ray_depths =
            ray_system.colPivHouseholderQr().solve(-translation_10);
        const Eigen::Vector3d point1_from_left =
            rotation_10 * (ray_depths.x() * left_ray) + translation_10;
        const Eigen::Vector3d point1_from_right =
            ray_depths.y() * right_ray;
        const Eigen::Vector3d point1 =
            0.5 * (point1_from_left + point1_from_right);
        const Eigen::Vector3d point0 =
            rotation_10.transpose() * (point1 - translation_10);
        if (!point0.allFinite() || !point1.allFinite() ||
            point0.z() < YOLOPOINT_LIGHTGLUE_STEREO_MIN_DEPTH ||
            point1.z() < YOLOPOINT_LIGHTGLUE_STEREO_MIN_DEPTH ||
            point0.z() > YOLOPOINT_LIGHTGLUE_STEREO_MAX_DEPTH ||
            point1.z() > YOLOPOINT_LIGHTGLUE_STEREO_MAX_DEPTH)
            continue;
        ++stereo_positive_depth_matches_;

        const Eigen::Vector2d left_reprojection =
            point0.head<2>() / point0.z();
        const Eigen::Vector2d right_reprojection =
            point1.head<2>() / point1.z();
        const double left_reprojection_error =
            (left_reprojection - left_ray.head<2>()).norm() * FOCAL_LENGTH;
        const double right_reprojection_error =
            (right_reprojection - right_ray.head<2>()).norm() * FOCAL_LENGTH;
        if (!std::isfinite(left_reprojection_error) ||
            !std::isfinite(right_reprojection_error) ||
            std::max(left_reprojection_error, right_reprojection_error) >
                YOLOPOINT_LIGHTGLUE_STEREO_REPROJECTION_THRESHOLD)
            continue;

        kept_indices.push_back(i);
        kept_right_points.push_back(right_points[i]);
    }

    vector<cv::Point2f> right_undistorted =
        undistort(kept_right_points, cameras_[1]);
    for (int i = 0; i < static_cast<int>(kept_indices.size()); ++i)
    {
        const Observation &left_observation =
            left_observations[kept_indices[i]];
        Observation right_observation;
        right_observation.id = left_observation.id;
        right_observation.track_count = left_observation.track_count;
        right_observation.keypoint =
            cv::KeyPoint(kept_right_points[i], 1.0f);
        right_observation.undistorted = right_undistorted[i];
        right_observation.velocity = velocityFor(
            right_observation.id, right_observation.undistorted,
            current_right_undistorted_map_,
            previous_right_undistorted_map_);
        right_observations[right_observation.id] = right_observation;
    }
    const int64 filter_end_ticks = cv::getTickCount();
    const double tick_ms = 1000.0 / cv::getTickFrequency();
    stereo_pyramid_ms_ =
        (flow_start_ticks - pyramid_start_ticks) * tick_ms;
    stereo_flow_ms_ =
        (filter_start_ticks - flow_start_ticks) * tick_ms;
    stereo_filter_ms_ =
        (filter_end_ticks - filter_start_ticks) * tick_ms;
}

cv::Mat YOLOPointLightGlueFeatureTracker::selectDescriptorRows(
    const cv::Mat &descriptors, const vector<int> &indices) const
{
    cv::Mat selected;
    for (int index : indices)
        selected.push_back(descriptors.row(index));
    return selected.clone();
}

cv::Rect2f YOLOPointLightGlueFeatureTracker::expandedDynamicBox(
    const Detection &detection) const
{
    const float margin = static_cast<float>(YOLOPOINT_DYNAMIC_BOX_MARGIN);
    const cv::Rect2f expanded(
        detection.box.x - margin, detection.box.y - margin,
        detection.box.width + 2.0f * margin,
        detection.box.height + 2.0f * margin);
    return expanded & cv::Rect2f(
        0.0f, 0.0f, static_cast<float>(col_), static_cast<float>(row_));
}

bool YOLOPointLightGlueFeatureTracker::pointInConfirmedDynamic(
    const cv::Point2f &point, const vector<Detection> &detections) const
{
    for (const Detection &detection : detections)
    {
        if (detection.dynamic_candidate && detection.moving &&
            expandedDynamicBox(detection).contains(point))
            return true;
    }
    return false;
}


void YOLOPointLightGlueFeatureTracker::classifyDynamicDetections(
    vector<Detection> &detections)
{
    // Dynamic filtering uses the detector prior directly.  No temporal
    // geometry, ROI tracking, or motion confirmation is involved.
    for (Detection &detection : detections)
    {
        detection.dynamic_candidate =
            isPotentiallyDynamicClass(detection.class_id);
        detection.geometry_evaluated = detection.dynamic_candidate;
        detection.moving = detection.dynamic_candidate &&
            YOLOPOINT_DYNAMIC_FEATURE_FILTER_ENABLE;
        detection.moving_pending = false;
        detection.geometry_tracks = 0;
        detection.moving_tracks = 0;
        detection.motion_evidence = 0;
        detection.static_evidence = 0;
        detection.moving_hold = 0;
    }
    return;

}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
YOLOPointLightGlueFeatureTracker::trackImage(
    double cur_time, const cv::Mat &left, const cv::Mat &right)
{
    current_time_ = cur_time;
    row_ = left.rows;
    col_ = left.cols;
    const int64 start_ticks = cv::getTickCount();

    ExtractedFeatures current = extractLeft(left);
    const int64 extraction_ticks = cv::getTickCount();
    vector<int> matched_ids;
    vector<int> matched_track_counts;
    vector<unsigned char> matched_status;
    matchTemporal(current, matched_ids, matched_track_counts, matched_status);
    map<int, cv::Point2f> temporal_geometry_points;
    trackTemporalGeometry(left, temporal_geometry_points);
    classifyDynamicDetections(current.detections);
    temporal_geometry_matches_ = 0;
    map<int, int> refreshed_model_indices;
    vector<unsigned char> used_current_model(current.keypoints.size(), 0);
    if (YOLOPOINT_LIGHTGLUE_USE_PERSISTENT_KLT_GEOMETRY)
    {
        for (int index = 0;
             index < static_cast<int>(matched_status.size()); ++index)
        {
            if (!matched_status[index])
                continue;
            const auto geometry =
                temporal_geometry_points.find(matched_ids[index]);
            if (geometry == temporal_geometry_points.end() ||
                cv::norm(
                    current.keypoints[index].pt - geometry->second) >
                    YOLOPOINT_LIGHTGLUE_MODEL_KLT_MAX_DISTANCE)
                continue;
            refreshed_model_indices[matched_ids[index]] = index;
            used_current_model[index] = 1;
            ++temporal_geometry_matches_;
        }
    }
    const int64 matching_ticks = cv::getTickCount();

    cv::Mat occupancy(row_, col_, CV_8UC1, cv::Scalar(255));
    vector<cv::Point2f> selected_points;
    vector<Observation> left_observations;
    dynamic_filtered_features_ = 0;
    const auto addObservation =
        [&](const Observation &observation)
    {
        if (static_cast<int>(left_observations.size()) >= MAX_CNT)
            return;
        const cv::Point2f point = observation.keypoint.pt;
        if (!inBorder(point, col_, row_))
            return;
        if (YOLOPOINT_DYNAMIC_FEATURE_FILTER_ENABLE &&
            pointInConfirmedDynamic(point, current.detections))
        {
            ++dynamic_filtered_features_;
            return;
        }
        const cv::Point rounded(cvRound(point.x), cvRound(point.y));
        if (occupancy.at<unsigned char>(rounded) == 0)
            return;
        left_observations.push_back(observation);
        selected_points.push_back(point);
        cv::circle(occupancy, point, MIN_DIST, 0, -1);
    };

    vector<int> current_order(current.keypoints.size());
    std::iota(current_order.begin(), current_order.end(), 0);
    if (YOLOPOINT_LIGHTGLUE_USE_PERSISTENT_KLT_GEOMETRY)
    {
        vector<Observation> persistent_observations;
        persistent_observations.reserve(previous_ids_.size());
        for (int previous_index = 0;
             previous_index < static_cast<int>(previous_ids_.size());
             ++previous_index)
        {
            const int id = previous_ids_[previous_index];
            const auto geometry = temporal_geometry_points.find(id);
            if (geometry == temporal_geometry_points.end())
                continue;
            Observation observation;
            observation.id = id;
            observation.track_count =
                previous_track_counts_[previous_index] + 1;
            observation.keypoint = cv::KeyPoint(geometry->second, 1.0f);

            const auto refreshed = refreshed_model_indices.find(id);
            if (refreshed != refreshed_model_indices.end())
            {
                observation.descriptor_index = refreshed->second;
                observation.model_keypoint =
                    current.keypoints[refreshed->second];
                observation.descriptor =
                    current.descriptors.row(refreshed->second).clone();
            }
            else
            {
                observation.model_keypoint =
                    previous_keypoints_[previous_index];
                observation.model_keypoint.pt = geometry->second;
                if (previous_index < previous_descriptors_.rows)
                    observation.descriptor =
                        previous_descriptors_.row(previous_index).clone();
            }
            if (!observation.descriptor.empty())
                persistent_observations.push_back(observation);
        }
        std::sort(
            persistent_observations.begin(),
            persistent_observations.end(),
            [](const Observation &a, const Observation &b)
            {
                if (a.track_count != b.track_count)
                    return a.track_count > b.track_count;
                return a.model_keypoint.response >
                       b.model_keypoint.response;
            });
        for (const Observation &observation : persistent_observations)
            addObservation(observation);

        std::sort(
            current_order.begin(), current_order.end(),
            [&](int a, int b)
            {
                return current.keypoints[a].response >
                       current.keypoints[b].response;
            });
        for (int index : current_order)
        {
            if (static_cast<int>(left_observations.size()) >= MAX_CNT)
                break;
            if (used_current_model[index])
                continue;
            Observation observation;
            observation.id = next_id_++;
            observation.track_count = 1;
            observation.descriptor_index = index;
            observation.descriptor =
                current.descriptors.row(index).clone();
            observation.model_keypoint = current.keypoints[index];
            observation.keypoint = current.keypoints[index];
            addObservation(observation);
        }
    }
    else
    {
        std::sort(current_order.begin(), current_order.end(), [&](int a, int b) {
            if (matched_status[a] != matched_status[b])
                return matched_status[a] > matched_status[b];
            if (matched_track_counts[a] != matched_track_counts[b])
                return matched_track_counts[a] > matched_track_counts[b];
            return current.keypoints[a].response >
                   current.keypoints[b].response;
        });
        for (int index : current_order)
        {
            if (static_cast<int>(left_observations.size()) >= MAX_CNT)
                break;
            Observation observation;
            observation.id =
                matched_status[index] ? matched_ids[index] : next_id_++;
            observation.track_count =
                matched_status[index] ? matched_track_counts[index] : 1;
            observation.descriptor_index = index;
            observation.descriptor =
                current.descriptors.row(index).clone();
            observation.model_keypoint = current.keypoints[index];
            observation.keypoint = current.keypoints[index];
            addObservation(observation);
        }
    }

    vector<cv::Point2f> left_undistorted;
    if (!selected_points.empty() && !cameras_.empty())
        left_undistorted = undistort(selected_points, cameras_[0]);
    current_undistorted_map_.clear();
    for (int i = 0; i < static_cast<int>(left_observations.size()); ++i)
    {
        left_observations[i].undistorted = left_undistorted[i];
        left_observations[i].velocity = velocityFor(
            left_observations[i].id,
            left_observations[i].undistorted,
            current_undistorted_map_,
            previous_undistorted_map_);
    }

    map<int, Observation> right_observations;
    if (!right.empty() && stereo_camera_)
        matchStereoKlt(left, right, left_observations, right_observations);
    else
        current_right_undistorted_map_.clear();
    const int64 stereo_ticks = cv::getTickCount();

    current_loop_features_ = LoopModelFeatures();
    if (LOOP_CLOSURE_ENABLE && LOOP_CLOSURE_MODE == 3)
    {
        current_loop_features_.global_descriptor = current.global_descriptor.clone();
        // Loop closure keeps the complete static model feature set for
        // long-baseline matching. VIO-selected observations are the only
        // entries carrying persistent feature IDs.
        vector<int> loop_indices;
        vector<int> original_to_loop(current.keypoints.size(), -1);
        loop_indices.reserve(current.keypoints.size());
        for (int index = 0;
             index < static_cast<int>(current.keypoints.size()); ++index)
        {
            if (YOLOPOINT_DYNAMIC_FEATURE_FILTER_ENABLE &&
                pointInConfirmedDynamic(
                    current.keypoints[index].pt, current.detections))
                continue;
            original_to_loop[index] = static_cast<int>(loop_indices.size());
            loop_indices.push_back(index);
            current_loop_features_.keypoints.push_back(current.keypoints[index]);
        }
        current_loop_features_.descriptors =
            selectDescriptorRows(current.descriptors, loop_indices);
        current_loop_features_.feature_ids.assign(
            loop_indices.size(), -1);
        current_loop_features_.right_points.assign(
            loop_indices.size(), cv::Point2f(0.0f, 0.0f));
        current_loop_features_.has_right.assign(loop_indices.size(), 0);
        for (const Observation &observation : left_observations)
        {
            const int model_index = observation.descriptor_index;
            if (model_index < 0 ||
                model_index >= static_cast<int>(current.keypoints.size()))
                continue;
            const int loop_index = original_to_loop[model_index];
            if (loop_index < 0)
                continue;
            current_loop_features_.feature_ids[loop_index] = observation.id;
            const auto right_it = right_observations.find(observation.id);
            if (right_it != right_observations.end())
            {
                current_loop_features_.right_points[loop_index] =
                    right_it->second.keypoint.pt;
                current_loop_features_.has_right[loop_index] = 1;
            }
        }
    }

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> feature_frame;
    for (const Observation &observation : left_observations)
    {
        Eigen::Matrix<double, 7, 1> measurement;
        measurement << observation.undistorted.x,
            observation.undistorted.y, 1.0,
            observation.keypoint.pt.x, observation.keypoint.pt.y,
            observation.velocity.x, observation.velocity.y;
        feature_frame[observation.id].emplace_back(0, measurement);

        auto right_it = right_observations.find(observation.id);
        if (right_it != right_observations.end())
        {
            const Observation &right_observation = right_it->second;
            Eigen::Matrix<double, 7, 1> right_measurement;
            right_measurement << right_observation.undistorted.x,
                right_observation.undistorted.y, 1.0,
                right_observation.keypoint.pt.x,
                right_observation.keypoint.pt.y,
                right_observation.velocity.x,
                right_observation.velocity.y;
            feature_frame[observation.id].emplace_back(1, right_measurement);
        }
    }

    const bool static_features_sufficient =
        static_cast<int>(left_observations.size()) >=
        YOLOPOINT_DYNAMIC_MIN_STATIC_FEATURES;
    if (YOLOPOINT_DYNAMIC_FEATURE_FILTER_ENABLE &&
        static_features_sufficient != dynamic_static_features_sufficient_)
    {
        if (static_features_sufficient)
            ROS_INFO("YOLOPoint dynamic filtering recovered sufficient static features: %zu/%d",
                     left_observations.size(),
                     YOLOPOINT_DYNAMIC_MIN_STATIC_FEATURES);
        else
            ROS_WARN("YOLOPoint dynamic filtering has insufficient static features: %zu/%d",
                     left_observations.size(),
                     YOLOPOINT_DYNAMIC_MIN_STATIC_FEATURES);
    }
    dynamic_static_features_sufficient_ = static_features_sufficient;

    drawTrackImage(left, right, left_observations, right_observations,
                   current.detections);
    previous_descriptors_.release();
    previous_keypoints_.clear();
    previous_ids_.clear();
    previous_track_counts_.clear();
    previous_left_pixel_map_.clear();
    for (const Observation &observation : left_observations)
    {
        previous_descriptors_.push_back(observation.descriptor);
        previous_keypoints_.push_back(observation.model_keypoint);
        previous_ids_.push_back(observation.id);
        previous_track_counts_.push_back(observation.track_count);
        previous_left_pixel_map_[observation.id] = observation.keypoint.pt;
    }
    previous_undistorted_map_ = current_undistorted_map_;
    previous_right_undistorted_map_ = current_right_undistorted_map_;
    previous_left_image_ = toGray8(left).clone();
    previous_time_ = current_time_;

    if (YOLOPOINT_LIGHTGLUE_LOG_STATS)
    {
        const double tick_ms = 1000.0 / cv::getTickFrequency();
        ROS_INFO(
            "YOLOPoint+LightGlue: candidates %zu, temporal %zu/%d kept/raw, "
            "KLT geometry %d/%d/%d model/F/FB, selected %zu, "
            "stereo %zu/%d/%d/%d/%d final/depth/epi/corner/fb; objects %zu; "
            "dynamic %d filtered, static %s; "
            "extract %.3f ms [prep %.3f, infer %.3f, decode %.3f], "
            "match %.3f ms [prep %.3f, infer %.3f, post %.3f], "
            "select+stereo %.3f ms [pyr %.3f, flow %.3f, filter %.3f], "
            "total %.3f ms",
            current.keypoints.size(),
            std::count(matched_status.begin(), matched_status.end(), 1),
            matcher_raw_matches_,
            temporal_geometry_matches_, temporal_klt_f_matches_,
            temporal_klt_fb_matches_,
            left_observations.size(), right_observations.size(),
            stereo_positive_depth_matches_, stereo_epipolar_matches_,
            stereo_corner_matches_, stereo_fb_matches_,
            current.detections.size(),
            dynamic_filtered_features_,
            dynamic_static_features_sufficient_ ? "enough" : "low",
            (extraction_ticks - start_ticks) * tick_ms,
            extractor_preprocess_ms_, extractor_inference_ms_,
            extractor_decode_ms_,
            (matching_ticks - extraction_ticks) * tick_ms,
            matcher_prepare_ms_, matcher_inference_ms_,
            matcher_postprocess_ms_,
            (stereo_ticks - matching_ticks) * tick_ms,
            stereo_pyramid_ms_, stereo_flow_ms_, stereo_filter_ms_,
            (stereo_ticks - start_ticks) * tick_ms);
    }
    return feature_frame;
}

void YOLOPointLightGlueFeatureTracker::readIntrinsicParameter(
    const vector<string> &calib_files)
{
    cameras_.clear();
    for (const string &file : calib_files)
    {
        ROS_INFO("reading parameter of camera %s", file.c_str());
        cameras_.push_back(
            camodocal::CameraFactory::instance()->generateCameraFromYamlFile(file));
    }
    stereo_camera_ = calib_files.size() == 2;
    if (FEATURE_TRACKER_TYPE == 3)
        initializeModels();
}

void YOLOPointLightGlueFeatureTracker::setPrediction(
    map<int, Eigen::Vector3d> &predict_pts)
{
    (void)predict_pts;
}

void YOLOPointLightGlueFeatureTracker::removeOutliers(set<int> &remove_ids)
{
    if (remove_ids.empty())
        return;

    cv::Mat kept_descriptors;
    vector<cv::KeyPoint> kept_keypoints;
    vector<int> kept_ids;
    vector<int> kept_track_counts;
    for (int i = 0; i < static_cast<int>(previous_ids_.size()); ++i)
    {
        if (remove_ids.count(previous_ids_[i]))
            continue;
        kept_descriptors.push_back(previous_descriptors_.row(i));
        kept_keypoints.push_back(previous_keypoints_[i]);
        kept_ids.push_back(previous_ids_[i]);
        kept_track_counts.push_back(previous_track_counts_[i]);
    }
    previous_descriptors_ = kept_descriptors.clone();
    previous_keypoints_.swap(kept_keypoints);
    previous_ids_.swap(kept_ids);
    previous_track_counts_.swap(kept_track_counts);
    for (int id : remove_ids)
    {
        previous_undistorted_map_.erase(id);
        previous_right_undistorted_map_.erase(id);
        previous_left_pixel_map_.erase(id);
    }
}

void YOLOPointLightGlueFeatureTracker::drawTrackImage(
    const cv::Mat &left,
    const cv::Mat &right,
    const vector<Observation> &left_observations,
    const map<int, Observation> &right_observations,
    const vector<Detection> &detections)
{
    if (!SHOW_TRACK)
        return;
    cv::Mat left_bgr = toBgr(left);
    const int right_offset = left.cols;
    if (!right.empty() && stereo_camera_)
    {
        cv::Mat right_bgr = toBgr(right);
        cv::hconcat(left_bgr, right_bgr, track_image_);
    }
    else
    {
        track_image_ = left_bgr;
    }

    const vector<string> &class_names = cocoClassNames();
    const cv::Rect2f image_bounds(0.0f, 0.0f,
                                  static_cast<float>(left.cols),
                                  static_cast<float>(left.rows));
    for (const Detection &detection : detections)
    {
        const cv::Rect2f clipped = detection.box & image_bounds;
        if (clipped.width <= 1.0f || clipped.height <= 1.0f)
            continue;
        cv::Scalar color(
            64 + (37 * detection.class_id) % 192,
            64 + (17 * detection.class_id) % 192,
            64 + (29 * detection.class_id) % 192);
        if (YOLOPOINT_DYNAMIC_FEATURE_FILTER_ENABLE &&
            detection.dynamic_candidate)
        {
            if (detection.moving)
                color = cv::Scalar(0, 0, 255);
            else if (detection.moving_pending)
                color = cv::Scalar(0, 128, 255);
            else if (detection.geometry_evaluated)
                color = cv::Scalar(0, 200, 0);
            else
                color = cv::Scalar(0, 200, 255);
        }
        cv::rectangle(track_image_, clipped, color, 2);
        const string class_name =
            detection.class_id >= 0 &&
                    detection.class_id < static_cast<int>(class_names.size())
                ? class_names[detection.class_id]
                : "class_" + std::to_string(detection.class_id);
        char detail[64];
        if (YOLOPOINT_DYNAMIC_FEATURE_FILTER_ENABLE &&
            detection.dynamic_candidate)
        {
            const char *state = detection.moving
                ? "moving"
                : (detection.moving_pending
                       ? "pending"
                       : (detection.geometry_evaluated
                              ? "static"
                              : "unknown"));
            std::snprintf(
                detail, sizeof(detail), " %.2f %s %d/%d e%d",
                detection.confidence, state, detection.moving_tracks,
                detection.geometry_tracks, detection.motion_evidence);
        }
        else
        {
            std::snprintf(detail, sizeof(detail), " %.2f",
                          detection.confidence);
        }
        const string label = class_name + detail;
        int baseline = 0;
        const cv::Size label_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        const int label_x = std::max(0, cvRound(clipped.x));
        const int label_y = std::max(
            label_size.height + 3, cvRound(clipped.y));
        cv::rectangle(
            track_image_,
            cv::Rect(label_x, label_y - label_size.height - 3,
                     label_size.width + 4, label_size.height + 4),
            color, cv::FILLED);
        cv::putText(track_image_, label, cv::Point(label_x + 2, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    for (const Observation &observation : left_observations)
    {
        const double age = std::min(1.0, observation.track_count / 20.0);
        cv::circle(track_image_, observation.keypoint.pt, 2,
                   cv::Scalar(255 * (1.0 - age), 0, 255 * age), 2);
        auto previous = previous_left_pixel_map_.find(observation.id);
        if (previous != previous_left_pixel_map_.end())
            cv::arrowedLine(track_image_, observation.keypoint.pt,
                            previous->second, cv::Scalar(0, 255, 0),
                            1, 8, 0, 0.2);
        auto right_it = right_observations.find(observation.id);
        if (right_it != right_observations.end())
        {
            cv::Point2f right_point = right_it->second.keypoint.pt;
            right_point.x += right_offset;
            cv::circle(track_image_, right_point, 2,
                       cv::Scalar(0, 255, 0), 2);
            cv::line(track_image_, observation.keypoint.pt, right_point,
                     cv::Scalar(0, 180, 0), 1);
        }
    }
}

cv::Mat YOLOPointLightGlueFeatureTracker::getTrackImage()
{
    return track_image_;
}

LoopModelFeatures YOLOPointLightGlueFeatureTracker::getLoopFeatures() const
{
    LoopModelFeatures copy = current_loop_features_;
    copy.descriptors = current_loop_features_.descriptors.clone();
    copy.global_descriptor = current_loop_features_.global_descriptor.clone();
    return copy;
}

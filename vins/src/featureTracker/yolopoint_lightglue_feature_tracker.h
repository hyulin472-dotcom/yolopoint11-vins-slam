/*******************************************************
 * YOLOPointv11 + LightGlue frontend for VINS-Fusion ROS2.
 *
 * Only the current left image is processed by YOLOPoint. LightGlue matches
 * previous-left to current-left features. Stereo observations are obtained
 * with CPU forward/backward pyramidal LK and rectified epipolar constraints.
 *******************************************************/

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "../estimator/parameters.h"
#include "../loop_closure/loop_model_features.h"
#include "yolopoint_cuda_postprocess.h"

class YOLOPointLightGlueFeatureTracker
{
  public:
    YOLOPointLightGlueFeatureTracker();

    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(
        double cur_time, const cv::Mat &left, const cv::Mat &right = cv::Mat());

    void readIntrinsicParameter(const std::vector<std::string> &calib_files);
    void setPrediction(std::map<int, Eigen::Vector3d> &predict_pts);
    void removeOutliers(std::set<int> &remove_ids);
    cv::Mat getTrackImage();
    LoopModelFeatures getLoopFeatures() const;
    void shutdown();

  private:
    struct Detection
    {
        cv::Rect2f box;
        float confidence = 0.0f;
        int class_id = -1;
        bool dynamic_candidate = false;
        bool geometry_evaluated = false;
        bool moving = false;
        bool moving_pending = false;
        int geometry_tracks = 0;
        int moving_tracks = 0;
        int motion_evidence = 0;
        int static_evidence = 0;
        int moving_hold = 0;
    };

    struct ExtractedFeatures
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cv::Mat global_descriptor;
        std::vector<Detection> detections;
    };

    struct Observation
    {
        int id = -1;
        int track_count = 0;
        int descriptor_index = -1;
        cv::Mat descriptor;
        cv::KeyPoint model_keypoint;
        cv::KeyPoint keypoint;
        cv::Point2f undistorted = cv::Point2f(0, 0);
        cv::Point2f velocity = cv::Point2f(0, 0);
    };

    void initializeModels();
    std::unique_ptr<Ort::Session> createSession(
        const std::string &model_path,
        const std::string &cache_path,
        const std::unordered_map<std::string, std::string> &trt_profile);
    ExtractedFeatures extractLeft(const cv::Mat &image);
    void decodeExtractor(const float *semi,
                         const std::vector<int64_t> &semi_shape,
                         const float *dense_descriptors,
                         const std::vector<int64_t> &descriptor_shape,
                         float scale_x,
                         float scale_y,
                         ExtractedFeatures &features) const;
    void decodeDetections(const float *predictions,
                          const std::vector<int64_t> &shape,
                          float scale_x,
                          float scale_y,
                          std::vector<Detection> &detections) const;
    void matchTemporal(const ExtractedFeatures &current,
                       std::vector<int> &matched_ids,
                       std::vector<int> &matched_track_counts,
                       std::vector<unsigned char> &matched_status);
    void trackTemporalGeometry(
        const cv::Mat &current_left,
        std::map<int, cv::Point2f> &tracked_points,
        std::map<int, cv::Point2f> *raw_tracked_points = nullptr);
    void matchStereoKlt(const cv::Mat &left,
                        const cv::Mat &right,
                        const std::vector<Observation> &left_observations,
                        std::map<int, Observation> &right_observations);
    std::vector<cv::Point2f> undistort(
        const std::vector<cv::Point2f> &points,
        const camodocal::CameraPtr &camera) const;
    cv::Point2f velocityFor(
        int id,
        const cv::Point2f &undistorted,
        std::map<int, cv::Point2f> &current_map,
        const std::map<int, cv::Point2f> &previous_map) const;
    bool inBorder(const cv::Point2f &point, int width, int height) const;
    cv::Mat selectDescriptorRows(
        const cv::Mat &descriptors, const std::vector<int> &indices) const;
    cv::Rect2f expandedDynamicBox(const Detection &detection) const;
    bool pointInConfirmedDynamic(
        const cv::Point2f &point,
        const std::vector<Detection> &detections) const;
    void classifyDynamicDetections(
        std::vector<Detection> &detections);
    void drawTrackImage(
        const cv::Mat &left,
        const cv::Mat &right,
        const std::vector<Observation> &left_observations,
        const std::map<int, Observation> &right_observations,
        const std::vector<Detection> &detections);

    Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "VINSYOLOPointLightGlue"};
    Ort::AllocatorWithDefaultOptions ort_allocator_;
    std::unique_ptr<Ort::Session> extractor_session_;
    std::unique_ptr<Ort::Session> matcher_session_;
    std::unique_ptr<YOLOPointCudaPostprocessor> cuda_postprocessor_;
    bool initialized_ = false;
    bool stereo_camera_ = false;
    bool object_detection_available_ = false;

    // Diagnostics only; these timings do not affect tracker output.
    double extractor_preprocess_ms_ = 0.0;
    double extractor_inference_ms_ = 0.0;
    double extractor_decode_ms_ = 0.0;
    double matcher_prepare_ms_ = 0.0;
    double matcher_inference_ms_ = 0.0;
    double matcher_postprocess_ms_ = 0.0;
    int matcher_raw_matches_ = 0;
    double stereo_pyramid_ms_ = 0.0;
    double stereo_flow_ms_ = 0.0;
    double stereo_filter_ms_ = 0.0;
    int stereo_fb_matches_ = 0;
    int stereo_corner_matches_ = 0;
    int stereo_epipolar_matches_ = 0;
    int stereo_positive_depth_matches_ = 0;
    int temporal_klt_fb_matches_ = 0;
    int temporal_klt_f_matches_ = 0;
    int temporal_geometry_matches_ = 0;
    int dynamic_filtered_features_ = 0;
    bool dynamic_static_features_sufficient_ = true;

    int row_ = 0;
    int col_ = 0;
    int next_id_ = 0;
    double current_time_ = 0.0;
    double previous_time_ = -1.0;

    cv::Mat previous_descriptors_;
    cv::Mat previous_left_image_;
    std::vector<cv::KeyPoint> previous_keypoints_;
    std::vector<int> previous_ids_;
    std::vector<int> previous_track_counts_;
    std::map<int, cv::Point2f> previous_undistorted_map_;
    std::map<int, cv::Point2f> current_undistorted_map_;
    std::map<int, cv::Point2f> previous_right_undistorted_map_;
    std::map<int, cv::Point2f> current_right_undistorted_map_;
    std::map<int, cv::Point2f> previous_left_pixel_map_;

    std::vector<camodocal::CameraPtr> cameras_;
    cv::Mat track_image_;
    LoopModelFeatures current_loop_features_;
};

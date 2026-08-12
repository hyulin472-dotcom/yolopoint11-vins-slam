/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <stdexcept>
#include <opencv2/core/utils/filesystem.hpp>

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

int USE_GPU;
int USE_GPU_ACC_FLOW;
int USE_GPU_CERES;

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int USE_KITTI_Z_MOTION_PRIOR;
double KITTI_Z_MOTION_PRIOR_WEIGHT;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;
int FEATURE_TRACKER_TYPE;
std::string YOLOPOINT_LIGHTGLUE_EXTRACTOR_MODEL_PATH;
std::string YOLOPOINT_LIGHTGLUE_MATCHER_MODEL_PATH;
std::string YOLOPOINT_LIGHTGLUE_EXTRACTOR_TRT_CACHE_PATH;
std::string YOLOPOINT_LIGHTGLUE_MATCHER_TRT_CACHE_PATH;
std::string YOLOPOINT_FULL_MODEL_PATH;
std::string YOLOPOINT_FULL_TRT_CACHE_PATH;
int YOLOPOINT_LIGHTGLUE_USE_TENSORRT;
int YOLOPOINT_LIGHTGLUE_USE_CUDA;
int YOLOPOINT_LIGHTGLUE_LOG_STATS;
int YOLOPOINT_LIGHTGLUE_USE_PERSISTENT_KLT_GEOMETRY;
int YOLOPOINT_LIGHTGLUE_PERSISTENT_KLT_USE_FUNDAMENTAL_FILTER;
int YOLOPOINT_LIGHTGLUE_MAX_KEYPOINTS;
int YOLOPOINT_LIGHTGLUE_INPUT_WIDTH;
int YOLOPOINT_LIGHTGLUE_INPUT_HEIGHT;
int YOLOPOINT_LIGHTGLUE_NMS_RADIUS;
int YOLOPOINT_LIGHTGLUE_REMOVE_BORDERS;
double YOLOPOINT_LIGHTGLUE_SCORE_THRESHOLD;
int YOLOPOINT_OBJECT_DETECTION_ENABLE;
double YOLOPOINT_OBJECT_CONFIDENCE_THRESHOLD;
double YOLOPOINT_OBJECT_IOU_THRESHOLD;
int YOLOPOINT_OBJECT_MAX_DETECTIONS;
int YOLOPOINT_DYNAMIC_FEATURE_FILTER_ENABLE;
int YOLOPOINT_DYNAMIC_MIN_STATIC_FEATURES;
double YOLOPOINT_DYNAMIC_BOX_MARGIN;
double YOLOPOINT_LIGHTGLUE_MODEL_KLT_MAX_DISTANCE;
double YOLOPOINT_LIGHTGLUE_STEREO_MIN_CORNER_EIGENVALUE;
double YOLOPOINT_LIGHTGLUE_STEREO_FACTOR_RESIDUAL_SCALE;
double YOLOPOINT_LIGHTGLUE_STEREO_EPIPOLAR_THRESHOLD;
double YOLOPOINT_LIGHTGLUE_STEREO_REPROJECTION_THRESHOLD;
double YOLOPOINT_LIGHTGLUE_STEREO_MIN_DEPTH;
double YOLOPOINT_LIGHTGLUE_STEREO_MAX_DEPTH;
int ONLINE_DENSE_MAPPING_ENABLE;
int ONLINE_DENSE_MAPPING_USE_SUBMAP_MANAGER;
int ONLINE_DENSE_MAPPING_USE_SUBMAP_QUALITY_FILTER;
int ONLINE_DENSE_MAPPING_MIN_SUBMAP_POINTS;
double ONLINE_DENSE_MAPPING_MIN_VALID_DEPTH_RATIO;
double ONLINE_DENSE_MAPPING_MAX_DEPTH_STD;
int ONLINE_DENSE_MAPPING_USE_LOCAL_RADIUS_FILTER;
double ONLINE_DENSE_MAPPING_RADIUS_FILTER_RADIUS;
int ONLINE_DENSE_MAPPING_RADIUS_FILTER_MIN_NEIGHBORS;
int ONLINE_DENSE_MAPPING_USE_ACTIVE_SUBMAP_WINDOW;
int ONLINE_DENSE_MAPPING_MAX_ACTIVE_SUBMAPS;
double ONLINE_DENSE_MAPPING_ACTIVE_RADIUS;
int ONLINE_DENSE_MAPPING_USE_POSE_UPDATE_REASSEMBLY;
int ONLINE_DENSE_MAPPING_USE_OCCUPANCY_FUSION;
int ONLINE_DENSE_MAPPING_OCCUPANCY_MIN_HITS;
double ONLINE_DENSE_MAPPING_OCCUPANCY_PROB_HIT;
double ONLINE_DENSE_MAPPING_OCCUPANCY_THRESHOLD;
int ONLINE_DENSE_MAPPING_KEYFRAME_STRIDE;
int ONLINE_DENSE_MAPPING_PIXEL_STEP;
int ONLINE_DENSE_MAPPING_MAX_POINTS_PER_KEYFRAME;
int ONLINE_DENSE_MAPPING_MAX_TOTAL_POINTS;
double ONLINE_DENSE_MAPPING_MIN_DEPTH;
double ONLINE_DENSE_MAPPING_MAX_DEPTH;
double ONLINE_DENSE_MAPPING_VOXEL_SIZE;
double ONLINE_DENSE_MAPPING_RECTIFIED_FOCAL;
double ONLINE_DENSE_MAPPING_MIN_DISPARITY;
double ONLINE_DENSE_MAPPING_MAX_DISPARITY;
double ONLINE_DENSE_MAPPING_MIN_GRADIENT;
std::string ONLINE_DENSE_MAPPING_FOUNDATION_MODEL_PATH;
int ONLINE_DENSE_MAPPING_FOUNDATION_USE_TENSORRT;
int ONLINE_DENSE_MAPPING_FOUNDATION_USE_CUDA;
std::string ONLINE_DENSE_MAPPING_FOUNDATION_TRT_CACHE_PATH;
int ONLINE_DENSE_MAPPING_FOUNDATION_INPUT_WIDTH;
int ONLINE_DENSE_MAPPING_FOUNDATION_INPUT_HEIGHT;
std::string ONLINE_DENSE_MAPPING_TOPIC;
int LOOP_CLOSURE_ENABLE;
int LOOP_CLOSURE_MODE;
int LOOP_CLOSURE_MAX_FEATURES;
int LOOP_CLOSURE_KEYFRAME_STRIDE;
int LOOP_CLOSURE_STM_SIZE;
int LOOP_CLOSURE_MAX_DATABASE_SIZE;
int LOOP_CLOSURE_TEMPORAL_CONSISTENCY;
int LOOP_CLOSURE_CANDIDATE_WINDOW;
int LOOP_CLOSURE_MIN_MATCHES;
int LOOP_CLOSURE_MIN_INLIERS;
int LOOP_CLOSURE_MIN_F_INLIERS;
int LOOP_CLOSURE_MIN_GRID_CELLS;
int LOOP_CLOSURE_PNP_ITERATIONS;
int LOOP_CLOSURE_POSE_GRAPH_ITERATIONS;
int LOOP_CLOSURE_MAX_PENDING_KEYFRAMES;
int LOOP_CLOSURE_MIN_INTERVAL;
int LOOP_CLOSURE_NEIGHBOR_EDGES;
int LOOP_CLOSURE_ALLOW_STEREO_FALLBACK;
double LOOP_CLOSURE_APPEARANCE_THRESHOLD;
double LOOP_CLOSURE_MODEL_APPEARANCE_THRESHOLD;
double LOOP_CLOSURE_MATCH_RATIO;
double LOOP_CLOSURE_MIN_INLIER_RATIO;
double LOOP_CLOSURE_MIN_F_INLIER_RATIO;
double LOOP_CLOSURE_MIN_INLIER_SPREAD;
double LOOP_CLOSURE_F_RANSAC_THRESHOLD;
double LOOP_CLOSURE_PNP_REPROJECTION_ERROR;
double LOOP_CLOSURE_MIN_DEPTH;
double LOOP_CLOSURE_MAX_DEPTH;
double LOOP_CLOSURE_STEREO_MAX_ERROR;
double LOOP_CLOSURE_MAX_TRANSLATION;
double LOOP_CLOSURE_MAX_ROTATION_DEG;
double LOOP_CLOSURE_ODOM_TRANSLATION_WEIGHT;
double LOOP_CLOSURE_ODOM_ROTATION_WEIGHT;
double LOOP_CLOSURE_LOOP_TRANSLATION_WEIGHT;
double LOOP_CLOSURE_LOOP_ROTATION_WEIGHT;
double LOOP_CLOSURE_MIN_TRANSLATION_STD;
double LOOP_CLOSURE_MAX_TRANSLATION_STD;
double LOOP_CLOSURE_MIN_ROTATION_STD_DEG;
double LOOP_CLOSURE_MAX_ROTATION_STD_DEG;
double LOOP_CLOSURE_GRAPH_MAX_TRANSLATION_ERROR;
double LOOP_CLOSURE_GRAPH_MAX_ROTATION_ERROR_DEG;
double LOOP_CLOSURE_GRAPH_MAX_NORMALIZED_ERROR;
double LOOP_CLOSURE_DUAL_MAX_TRANSLATION_ERROR;
double LOOP_CLOSURE_DUAL_MAX_ROTATION_ERROR_DEG;
double LOOP_CLOSURE_GRAVITY_WEIGHT;

std::string WORLD_FRAME_ID;
std::string BODY_FRAME_ID;
std::string CAMERA_FRAME_ID;

namespace
{
bool fileExists(const std::string &path)
{
    FILE *fp = fopen(path.c_str(), "r");
    if (fp == nullptr)
        return false;
    fclose(fp);
    return true;
}

bool isAbsolutePath(const std::string &path)
{
    return !path.empty() && path[0] == '/';
}

int readBoolParam(const cv::FileNode &node, int default_value)
{
    if (node.empty())
        return default_value;
    if (node.isString())
    {
        std::string value = static_cast<std::string>(node);
        size_t comment_pos = value.find('#');
        if (comment_pos != std::string::npos)
            value = value.substr(0, comment_pos);
        value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        return value == "true" || value == "1" || value == "yes";
    }
    if (node.isInt() || node.isReal())
        return static_cast<int>(node) != 0;
    ROS_WARN("invalid boolean YAML value for %s, using default %s",
             node.name().c_str(), default_value ? "true" : "false");
    return default_value;
}

double readDoubleParam(const cv::FileNode &node, double default_value)
{
    if (node.empty())
        return default_value;
    if (node.isString())
    {
        std::string value = static_cast<std::string>(node);
        size_t comment_pos = value.find('#');
        if (comment_pos != std::string::npos)
            value = value.substr(0, comment_pos);
        value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
        if (value.empty())
            return default_value;
        return std::stod(value);
    }
    return static_cast<double>(node);
}
}

template <typename T>
T readParam(rclcpp::Node::SharedPtr n, std::string name)
{
    T ans;
    if (n->get_parameter(name, ans))
    {
        ROS_INFO("Loaded %s: ", name);
        std::cout << ans << std::endl;
    }
    else
    {
        ROS_ERROR("Failed to load %s", name);
        rclcpp::shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        // ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    int config_dir_pos_for_online = config_file.find_last_of('/');
    std::string configPathForOnline = config_dir_pos_for_online == -1 ? "." : config_file.substr(0, config_dir_pos_for_online);
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];
    FEATURE_TRACKER_TYPE = fsSettings["feature_tracker_type"].empty() ? 0 : static_cast<int>(fsSettings["feature_tracker_type"]);
    if (FEATURE_TRACKER_TYPE == 0)
        ROS_INFO("feature tracker type: LK optical flow");
    else if (FEATURE_TRACKER_TYPE == 3)
        ROS_INFO("feature tracker type: YOLOPointv11 + LightGlue temporal matching + stereo KLT");
    else
    {
        ROS_WARN("feature_tracker_type supports only 0 (LK) or 3 (YOLOPoint), got %d; fallback to LK optical flow",
                 FEATURE_TRACKER_TYPE);
        FEATURE_TRACKER_TYPE = 0;
    }

    std::string yolopoint_profile = "legacy";
    if (!fsSettings["yolopoint_lightglue_profile"].empty())
        fsSettings["yolopoint_lightglue_profile"] >> yolopoint_profile;
    yolopoint_profile.erase(
        std::remove_if(yolopoint_profile.begin(), yolopoint_profile.end(), ::isspace),
        yolopoint_profile.end());
    std::transform(yolopoint_profile.begin(), yolopoint_profile.end(),
                   yolopoint_profile.begin(), ::tolower);
    if (yolopoint_profile != "legacy" && yolopoint_profile != "indoor" &&
        yolopoint_profile != "outdoor")
    {
        ROS_WARN("yolopoint_lightglue_profile must be indoor or outdoor, got '%s'; using indoor",
                 yolopoint_profile.c_str());
        yolopoint_profile = "indoor";
    }
    const bool yolopoint_profile_indoor = yolopoint_profile == "indoor";
    const bool yolopoint_profile_outdoor = yolopoint_profile == "outdoor";

    std::string yolopoint_backend = "legacy";
    if (!fsSettings["yolopoint_lightglue_backend"].empty())
        fsSettings["yolopoint_lightglue_backend"] >> yolopoint_backend;
    yolopoint_backend.erase(
        std::remove_if(yolopoint_backend.begin(), yolopoint_backend.end(), ::isspace),
        yolopoint_backend.end());
    std::transform(yolopoint_backend.begin(), yolopoint_backend.end(),
                   yolopoint_backend.begin(), ::tolower);
    if (yolopoint_backend != "legacy" && yolopoint_backend != "tensorrt" &&
        yolopoint_backend != "cuda" && yolopoint_backend != "cpu")
    {
        ROS_WARN("yolopoint_lightglue_backend must be tensorrt, cuda or cpu, got '%s'; using cuda",
                 yolopoint_backend.c_str());
        yolopoint_backend = "cuda";
    }

    fsSettings["yolopoint_lightglue_extractor_model_path"] >> YOLOPOINT_LIGHTGLUE_EXTRACTOR_MODEL_PATH;
    if (YOLOPOINT_LIGHTGLUE_EXTRACTOR_MODEL_PATH.empty())
        YOLOPOINT_LIGHTGLUE_EXTRACTOR_MODEL_PATH =
            "onxx/yolopointv11_lightglue/yolopointv11_extractor.onnx";
    if (!isAbsolutePath(YOLOPOINT_LIGHTGLUE_EXTRACTOR_MODEL_PATH) &&
        !fileExists(YOLOPOINT_LIGHTGLUE_EXTRACTOR_MODEL_PATH))
    {
        std::string project_relative =
            configPathForOnline + "/../../" + YOLOPOINT_LIGHTGLUE_EXTRACTOR_MODEL_PATH;
        if (fileExists(project_relative))
            YOLOPOINT_LIGHTGLUE_EXTRACTOR_MODEL_PATH = project_relative;
    }

    fsSettings["yolopoint_lightglue_matcher_model_path"] >> YOLOPOINT_LIGHTGLUE_MATCHER_MODEL_PATH;
    if (YOLOPOINT_LIGHTGLUE_MATCHER_MODEL_PATH.empty())
        YOLOPOINT_LIGHTGLUE_MATCHER_MODEL_PATH =
            "onxx/yolopointv11_lightglue/yolopointv11_lightglue.onnx";
    if (!isAbsolutePath(YOLOPOINT_LIGHTGLUE_MATCHER_MODEL_PATH) &&
        !fileExists(YOLOPOINT_LIGHTGLUE_MATCHER_MODEL_PATH))
    {
        std::string project_relative =
            configPathForOnline + "/../../" + YOLOPOINT_LIGHTGLUE_MATCHER_MODEL_PATH;
        if (fileExists(project_relative))
            YOLOPOINT_LIGHTGLUE_MATCHER_MODEL_PATH = project_relative;
    }

    fsSettings["yolopoint_lightglue_extractor_trt_cache_path"] >>
        YOLOPOINT_LIGHTGLUE_EXTRACTOR_TRT_CACHE_PATH;
    if (YOLOPOINT_LIGHTGLUE_EXTRACTOR_TRT_CACHE_PATH.empty())
        YOLOPOINT_LIGHTGLUE_EXTRACTOR_TRT_CACHE_PATH = yolopoint_profile_outdoor
            ? "onxx/yolopointv11_lightglue/trt_cache/extractor_kitti_1248x384"
            : "onxx/yolopointv11_lightglue/trt_cache/extractor";
    if (!isAbsolutePath(YOLOPOINT_LIGHTGLUE_EXTRACTOR_TRT_CACHE_PATH))
        YOLOPOINT_LIGHTGLUE_EXTRACTOR_TRT_CACHE_PATH =
            configPathForOnline + "/../../" + YOLOPOINT_LIGHTGLUE_EXTRACTOR_TRT_CACHE_PATH;

    fsSettings["yolopoint_full_model_path"] >> YOLOPOINT_FULL_MODEL_PATH;
    if (YOLOPOINT_FULL_MODEL_PATH.empty())
        YOLOPOINT_FULL_MODEL_PATH =
            "onxx/yolopointv11_lightglue/yolopointv11_full.onnx";
    if (!isAbsolutePath(YOLOPOINT_FULL_MODEL_PATH) &&
        !fileExists(YOLOPOINT_FULL_MODEL_PATH))
    {
        std::string project_relative =
            configPathForOnline + "/../../" + YOLOPOINT_FULL_MODEL_PATH;
        if (fileExists(project_relative))
            YOLOPOINT_FULL_MODEL_PATH = project_relative;
    }

    fsSettings["yolopoint_full_trt_cache_path"] >>
        YOLOPOINT_FULL_TRT_CACHE_PATH;
    if (YOLOPOINT_FULL_TRT_CACHE_PATH.empty())
        YOLOPOINT_FULL_TRT_CACHE_PATH =
            "onxx/yolopointv11_lightglue/trt_cache/full_extractor_dynamic";
    if (!isAbsolutePath(YOLOPOINT_FULL_TRT_CACHE_PATH))
        YOLOPOINT_FULL_TRT_CACHE_PATH =
            configPathForOnline + "/../../" + YOLOPOINT_FULL_TRT_CACHE_PATH;

    fsSettings["yolopoint_lightglue_matcher_trt_cache_path"] >>
        YOLOPOINT_LIGHTGLUE_MATCHER_TRT_CACHE_PATH;
    if (YOLOPOINT_LIGHTGLUE_MATCHER_TRT_CACHE_PATH.empty())
        YOLOPOINT_LIGHTGLUE_MATCHER_TRT_CACHE_PATH =
            "onxx/yolopointv11_lightglue/trt_cache/lightglue_dynamic_n1_512_opt256";
    if (!isAbsolutePath(YOLOPOINT_LIGHTGLUE_MATCHER_TRT_CACHE_PATH))
        YOLOPOINT_LIGHTGLUE_MATCHER_TRT_CACHE_PATH =
            configPathForOnline + "/../../" + YOLOPOINT_LIGHTGLUE_MATCHER_TRT_CACHE_PATH;

    YOLOPOINT_LIGHTGLUE_USE_TENSORRT =
        readBoolParam(fsSettings["yolopoint_lightglue_use_tensorrt"], 1);
    YOLOPOINT_LIGHTGLUE_USE_CUDA =
        readBoolParam(fsSettings["yolopoint_lightglue_use_cuda"], 1);
    if (yolopoint_backend == "tensorrt")
    {
        YOLOPOINT_LIGHTGLUE_USE_TENSORRT = 1;
    }
    else if (yolopoint_backend == "cuda")
    {
        YOLOPOINT_LIGHTGLUE_USE_TENSORRT = 0;
        YOLOPOINT_LIGHTGLUE_USE_CUDA = 1;
    }
    else if (yolopoint_backend == "cpu")
    {
        YOLOPOINT_LIGHTGLUE_USE_TENSORRT = 0;
        YOLOPOINT_LIGHTGLUE_USE_CUDA = 0;
    }
#if !VINS_HAS_CUDA
    if (YOLOPOINT_LIGHTGLUE_USE_CUDA)
        ROS_WARN("CUDA postprocessing requested but unavailable in this build; using CPU postprocessing");
    YOLOPOINT_LIGHTGLUE_USE_CUDA = 0;
#endif
    YOLOPOINT_LIGHTGLUE_LOG_STATS =
        readBoolParam(fsSettings["yolopoint_lightglue_log_stats"], 0);
    YOLOPOINT_LIGHTGLUE_USE_PERSISTENT_KLT_GEOMETRY =
        readBoolParam(
            fsSettings[
                "yolopoint_lightglue_use_persistent_klt_geometry"],
            (yolopoint_profile_indoor || yolopoint_profile_outdoor) ? 1 : 0);
    YOLOPOINT_LIGHTGLUE_PERSISTENT_KLT_USE_FUNDAMENTAL_FILTER =
        readBoolParam(
            fsSettings[
                "yolopoint_lightglue_persistent_klt_use_fundamental_filter"],
            yolopoint_profile_outdoor ? 1 : 0);
    YOLOPOINT_LIGHTGLUE_MAX_KEYPOINTS = std::max(
        1, std::min(512, static_cast<int>(readDoubleParam(
                            fsSettings["yolopoint_lightglue_max_keypoints"], 512.0))));
    YOLOPOINT_LIGHTGLUE_INPUT_WIDTH = std::max(
        32, static_cast<int>(readDoubleParam(
                fsSettings["yolopoint_lightglue_input_width"],
                yolopoint_profile_outdoor ? 1248.0 : 752.0)));
    YOLOPOINT_LIGHTGLUE_INPUT_HEIGHT = std::max(
        32, static_cast<int>(readDoubleParam(
                fsSettings["yolopoint_lightglue_input_height"],
                yolopoint_profile_outdoor ? 384.0 : 480.0)));
    YOLOPOINT_LIGHTGLUE_NMS_RADIUS = std::max(
        0, static_cast<int>(readDoubleParam(
               fsSettings["yolopoint_lightglue_nms_radius"], 4.0)));
    YOLOPOINT_LIGHTGLUE_REMOVE_BORDERS = std::max(
        0, static_cast<int>(readDoubleParam(
               fsSettings["yolopoint_lightglue_remove_borders"], 4.0)));
    YOLOPOINT_LIGHTGLUE_SCORE_THRESHOLD =
        readDoubleParam(fsSettings["yolopoint_lightglue_score_threshold"], 0.015);
    YOLOPOINT_OBJECT_DETECTION_ENABLE =
        readBoolParam(fsSettings["yolopoint_object_detection_enable"], 0);
    if (YOLOPOINT_OBJECT_DETECTION_ENABLE)
    {
        YOLOPOINT_LIGHTGLUE_INPUT_WIDTH = std::max(
            32, static_cast<int>(readDoubleParam(
                    fsSettings["yolopoint_full_input_width"],
                    yolopoint_profile_outdoor ? 1248.0 : 768.0)));
        YOLOPOINT_LIGHTGLUE_INPUT_HEIGHT = std::max(
            32, static_cast<int>(readDoubleParam(
                    fsSettings["yolopoint_full_input_height"],
                    yolopoint_profile_outdoor ? 384.0 : 480.0)));
        if (YOLOPOINT_LIGHTGLUE_INPUT_WIDTH % 32 ||
            YOLOPOINT_LIGHTGLUE_INPUT_HEIGHT % 32)
            throw std::runtime_error(
                "full YOLOPoint input width and height must be divisible by 32");
    }
    YOLOPOINT_OBJECT_CONFIDENCE_THRESHOLD = std::min(
        1.0, std::max(0.0, readDoubleParam(
            fsSettings["yolopoint_object_confidence_threshold"], 0.25)));
    YOLOPOINT_OBJECT_IOU_THRESHOLD = std::min(
        1.0, std::max(0.0, readDoubleParam(
            fsSettings["yolopoint_object_iou_threshold"], 0.45)));
    YOLOPOINT_OBJECT_MAX_DETECTIONS = std::max(
        1, static_cast<int>(readDoubleParam(
               fsSettings["yolopoint_object_max_detections"], 100.0)));
    const int dynamic_filter_requested = readBoolParam(
        fsSettings["yolopoint_dynamic_feature_filter_enable"], 0);
    YOLOPOINT_DYNAMIC_FEATURE_FILTER_ENABLE =
        dynamic_filter_requested && FEATURE_TRACKER_TYPE == 3 &&
        YOLOPOINT_OBJECT_DETECTION_ENABLE;
    if (dynamic_filter_requested && !YOLOPOINT_DYNAMIC_FEATURE_FILTER_ENABLE)
        ROS_WARN(
            "YOLOPoint dynamic filtering requires feature_tracker_type: 3 "
            "and yolopoint_object_detection_enable: true; ignoring request");
    YOLOPOINT_DYNAMIC_MIN_STATIC_FEATURES = std::max(
        0, static_cast<int>(readDoubleParam(
               fsSettings["yolopoint_dynamic_min_static_features"], 40.0)));
    YOLOPOINT_DYNAMIC_BOX_MARGIN = std::max(
        0.0, readDoubleParam(fsSettings["yolopoint_dynamic_box_margin"], 3.0));
    YOLOPOINT_LIGHTGLUE_MODEL_KLT_MAX_DISTANCE =
        std::max(0.0, readDoubleParam(
            fsSettings["yolopoint_lightglue_model_klt_max_distance"], 1.0));
    YOLOPOINT_LIGHTGLUE_STEREO_MIN_CORNER_EIGENVALUE =
        std::max(0.0, readDoubleParam(
            fsSettings[
                "yolopoint_lightglue_stereo_min_corner_eigenvalue"],
            (yolopoint_profile_indoor || yolopoint_profile_outdoor) ? 0.01 : 0.0));
    YOLOPOINT_LIGHTGLUE_STEREO_FACTOR_RESIDUAL_SCALE =
        std::max(0.0, readDoubleParam(
            fsSettings[
                "yolopoint_lightglue_stereo_factor_residual_scale"],
            (yolopoint_profile_indoor || yolopoint_profile_outdoor) ? 0.25 : 1.0));
    YOLOPOINT_LIGHTGLUE_STEREO_EPIPOLAR_THRESHOLD =
        std::max(0.0, readDoubleParam(
            fsSettings["yolopoint_lightglue_stereo_epipolar_threshold"], 1.5));
    YOLOPOINT_LIGHTGLUE_STEREO_REPROJECTION_THRESHOLD =
        std::max(0.0, readDoubleParam(
            fsSettings["yolopoint_lightglue_stereo_reprojection_threshold"], 1.5));
    YOLOPOINT_LIGHTGLUE_STEREO_MIN_DEPTH =
        std::max(0.0, readDoubleParam(
            fsSettings["yolopoint_lightglue_stereo_min_depth"],
            yolopoint_profile_outdoor ? 1.0 : 0.1));
    YOLOPOINT_LIGHTGLUE_STEREO_MAX_DEPTH =
        std::max(YOLOPOINT_LIGHTGLUE_STEREO_MIN_DEPTH,
                 readDoubleParam(
                     fsSettings["yolopoint_lightglue_stereo_max_depth"], 200.0));
    ROS_INFO("YOLOPoint+LightGlue: profile %s, backend %s, TensorRT %s, CUDA EP %s, candidates %d, input %dx%d",
             yolopoint_profile.c_str(), yolopoint_backend.c_str(),
             YOLOPOINT_LIGHTGLUE_USE_TENSORRT ? "true" : "false",
             YOLOPOINT_LIGHTGLUE_USE_CUDA ? "true" : "false",
             YOLOPOINT_LIGHTGLUE_MAX_KEYPOINTS,
             YOLOPOINT_LIGHTGLUE_INPUT_WIDTH,
             YOLOPOINT_LIGHTGLUE_INPUT_HEIGHT);
    ROS_INFO(
        "YOLOPoint object detection: %s, confidence %.2f, IoU %.2f, max %d",
        YOLOPOINT_OBJECT_DETECTION_ENABLE ? "enabled" : "disabled",
        YOLOPOINT_OBJECT_CONFIDENCE_THRESHOLD,
        YOLOPOINT_OBJECT_IOU_THRESHOLD,
        YOLOPOINT_OBJECT_MAX_DETECTIONS);
    ROS_INFO(
        "YOLOPoint dynamic filtering: %s, direct detection-box prior, "
        "minimum static features %d, box margin %.1f px",
        YOLOPOINT_DYNAMIC_FEATURE_FILTER_ENABLE ? "enabled" : "disabled",
        YOLOPOINT_DYNAMIC_MIN_STATIC_FEATURES,
        YOLOPOINT_DYNAMIC_BOX_MARGIN);
    ROS_INFO(
        "YOLOPoint+LightGlue persistent KLT geometry: %s, "
        "temporal F filter: %s, model-KLT gate %.2f px",
        YOLOPOINT_LIGHTGLUE_USE_PERSISTENT_KLT_GEOMETRY
            ? "true"
            : "false",
        YOLOPOINT_LIGHTGLUE_PERSISTENT_KLT_USE_FUNDAMENTAL_FILTER
            ? "true"
            : "false",
        YOLOPOINT_LIGHTGLUE_MODEL_KLT_MAX_DISTANCE);
    ROS_INFO(
        "YOLOPoint+LightGlue stereo geometry: epipolar %.2f px, "
        "reprojection %.2f px, depth [%.2f, %.2f] m, "
        "min corner eigenvalue %.6f, factor residual scale %.3f",
        YOLOPOINT_LIGHTGLUE_STEREO_EPIPOLAR_THRESHOLD,
        YOLOPOINT_LIGHTGLUE_STEREO_REPROJECTION_THRESHOLD,
        YOLOPOINT_LIGHTGLUE_STEREO_MIN_DEPTH,
        YOLOPOINT_LIGHTGLUE_STEREO_MAX_DEPTH,
        YOLOPOINT_LIGHTGLUE_STEREO_MIN_CORNER_EIGENVALUE,
        YOLOPOINT_LIGHTGLUE_STEREO_FACTOR_RESIDUAL_SCALE);

    ONLINE_DENSE_MAPPING_ENABLE = readBoolParam(fsSettings["online_dense_mapping_enable"], 0);
    ONLINE_DENSE_MAPPING_USE_SUBMAP_MANAGER = readBoolParam(fsSettings["online_dense_mapping_use_submap_manager"], 0);
    ONLINE_DENSE_MAPPING_USE_SUBMAP_QUALITY_FILTER =
        readBoolParam(fsSettings["online_dense_mapping_use_submap_quality_filter"], 0);
    ONLINE_DENSE_MAPPING_MIN_SUBMAP_POINTS =
        std::max(1, static_cast<int>(readDoubleParam(fsSettings["online_dense_mapping_min_submap_points"], 300.0)));
    ONLINE_DENSE_MAPPING_MIN_VALID_DEPTH_RATIO =
        readDoubleParam(fsSettings["online_dense_mapping_min_valid_depth_ratio"], 0.02);
    ONLINE_DENSE_MAPPING_MAX_DEPTH_STD =
        readDoubleParam(fsSettings["online_dense_mapping_max_depth_std"], 8.0);
    ONLINE_DENSE_MAPPING_USE_LOCAL_RADIUS_FILTER =
        readBoolParam(fsSettings["online_dense_mapping_use_local_radius_filter"], 0);
    ONLINE_DENSE_MAPPING_RADIUS_FILTER_RADIUS =
        readDoubleParam(fsSettings["online_dense_mapping_radius_filter_radius"], 0.25);
    ONLINE_DENSE_MAPPING_RADIUS_FILTER_MIN_NEIGHBORS =
        std::max(1, static_cast<int>(readDoubleParam(fsSettings["online_dense_mapping_radius_filter_min_neighbors"], 2.0)));
    ONLINE_DENSE_MAPPING_USE_ACTIVE_SUBMAP_WINDOW =
        readBoolParam(fsSettings["online_dense_mapping_use_active_submap_window"], 0);
    ONLINE_DENSE_MAPPING_MAX_ACTIVE_SUBMAPS =
        std::max(1, static_cast<int>(readDoubleParam(fsSettings["online_dense_mapping_max_active_submaps"], 120.0)));
    ONLINE_DENSE_MAPPING_ACTIVE_RADIUS =
        readDoubleParam(fsSettings["online_dense_mapping_active_radius"], 80.0);
    ONLINE_DENSE_MAPPING_USE_POSE_UPDATE_REASSEMBLY =
        readBoolParam(fsSettings["online_dense_mapping_use_pose_update_reassembly"], 0);
    ONLINE_DENSE_MAPPING_USE_OCCUPANCY_FUSION =
        readBoolParam(fsSettings["online_dense_mapping_use_occupancy_fusion"], 0);
    ONLINE_DENSE_MAPPING_OCCUPANCY_MIN_HITS =
        std::max(1, static_cast<int>(readDoubleParam(fsSettings["online_dense_mapping_occupancy_min_hits"], 2.0)));
    ONLINE_DENSE_MAPPING_OCCUPANCY_PROB_HIT =
        std::min(0.99, std::max(0.51, readDoubleParam(fsSettings["online_dense_mapping_occupancy_prob_hit"], 0.70)));
    ONLINE_DENSE_MAPPING_OCCUPANCY_THRESHOLD =
        std::min(0.99, std::max(0.01, readDoubleParam(fsSettings["online_dense_mapping_occupancy_threshold"], 0.65)));
    ONLINE_DENSE_MAPPING_KEYFRAME_STRIDE =
        std::max(1, static_cast<int>(readDoubleParam(fsSettings["online_dense_mapping_keyframe_stride"], 1.0)));
    ONLINE_DENSE_MAPPING_PIXEL_STEP =
        std::max(1, static_cast<int>(readDoubleParam(fsSettings["online_dense_mapping_pixel_step"], 6.0)));
    ONLINE_DENSE_MAPPING_MAX_POINTS_PER_KEYFRAME =
        std::max(1, static_cast<int>(readDoubleParam(fsSettings["online_dense_mapping_max_points_per_keyframe"], 2500.0)));
    ONLINE_DENSE_MAPPING_MAX_TOTAL_POINTS =
        static_cast<int>(readDoubleParam(fsSettings["online_dense_mapping_max_total_points"], 300000.0));
    ONLINE_DENSE_MAPPING_MIN_DEPTH = readDoubleParam(fsSettings["online_dense_mapping_min_depth"], 0.3);
    ONLINE_DENSE_MAPPING_MAX_DEPTH = readDoubleParam(fsSettings["online_dense_mapping_max_depth"], 15.0);
    ONLINE_DENSE_MAPPING_VOXEL_SIZE = readDoubleParam(fsSettings["online_dense_mapping_voxel_size"], 0.08);
    ONLINE_DENSE_MAPPING_RECTIFIED_FOCAL = readDoubleParam(fsSettings["online_dense_mapping_rectified_focal"], 460.0);
    ONLINE_DENSE_MAPPING_MIN_DISPARITY = readDoubleParam(fsSettings["online_dense_mapping_min_disparity"], 2.0);
    ONLINE_DENSE_MAPPING_MAX_DISPARITY = readDoubleParam(fsSettings["online_dense_mapping_max_disparity"], 96.0);
    ONLINE_DENSE_MAPPING_MIN_GRADIENT = readDoubleParam(fsSettings["online_dense_mapping_min_gradient"], 12.0);
    fsSettings["online_dense_mapping_foundation_model_path"] >> ONLINE_DENSE_MAPPING_FOUNDATION_MODEL_PATH;
    if (ONLINE_DENSE_MAPPING_FOUNDATION_MODEL_PATH.empty())
        ONLINE_DENSE_MAPPING_FOUNDATION_MODEL_PATH = "onxx/fast_foundationstereo/20_30_48_iters_4_res_320x736.onnx";
    if (!isAbsolutePath(ONLINE_DENSE_MAPPING_FOUNDATION_MODEL_PATH) && !fileExists(ONLINE_DENSE_MAPPING_FOUNDATION_MODEL_PATH))
    {
        std::string project_relative = configPathForOnline + "/../../" + ONLINE_DENSE_MAPPING_FOUNDATION_MODEL_PATH;
        if (fileExists(project_relative))
            ONLINE_DENSE_MAPPING_FOUNDATION_MODEL_PATH = project_relative;
    }
    ONLINE_DENSE_MAPPING_FOUNDATION_USE_TENSORRT =
        readBoolParam(fsSettings["online_dense_mapping_foundation_use_tensorrt"], 0);
    ONLINE_DENSE_MAPPING_FOUNDATION_USE_CUDA = readBoolParam(fsSettings["online_dense_mapping_foundation_use_cuda"], 1);
    fsSettings["online_dense_mapping_foundation_trt_cache_path"] >> ONLINE_DENSE_MAPPING_FOUNDATION_TRT_CACHE_PATH;
    if (ONLINE_DENSE_MAPPING_FOUNDATION_TRT_CACHE_PATH.empty())
        ONLINE_DENSE_MAPPING_FOUNDATION_TRT_CACHE_PATH = "onxx/fast_foundationstereo/trt_cache";
    if (!isAbsolutePath(ONLINE_DENSE_MAPPING_FOUNDATION_TRT_CACHE_PATH))
        ONLINE_DENSE_MAPPING_FOUNDATION_TRT_CACHE_PATH =
            configPathForOnline + "/../../" + ONLINE_DENSE_MAPPING_FOUNDATION_TRT_CACHE_PATH;
    ONLINE_DENSE_MAPPING_FOUNDATION_INPUT_WIDTH =
        std::max(1, static_cast<int>(readDoubleParam(fsSettings["online_dense_mapping_foundation_input_width"], 736.0)));
    ONLINE_DENSE_MAPPING_FOUNDATION_INPUT_HEIGHT =
        std::max(1, static_cast<int>(readDoubleParam(fsSettings["online_dense_mapping_foundation_input_height"], 320.0)));
    fsSettings["online_dense_mapping_topic"] >> ONLINE_DENSE_MAPPING_TOPIC;
    if (ONLINE_DENSE_MAPPING_TOPIC.empty())
        ONLINE_DENSE_MAPPING_TOPIC = "~/online_dense_cloud";
    ROS_INFO("online dense mapping: %s, submap manager: %s, quality filter: %s, active window: %s, pose reassembly: %s, occupancy fusion: %s, topic %s",
             ONLINE_DENSE_MAPPING_ENABLE ? "true" : "false",
             ONLINE_DENSE_MAPPING_USE_SUBMAP_MANAGER ? "true" : "false",
             ONLINE_DENSE_MAPPING_USE_SUBMAP_QUALITY_FILTER ? "true" : "false",
             ONLINE_DENSE_MAPPING_USE_ACTIVE_SUBMAP_WINDOW ? "true" : "false",
             ONLINE_DENSE_MAPPING_USE_POSE_UPDATE_REASSEMBLY ? "true" : "false",
             ONLINE_DENSE_MAPPING_USE_OCCUPANCY_FUSION ? "true" : "false",
             ONLINE_DENSE_MAPPING_TOPIC.c_str());

    // Independent loop closure backend. Mode 1 is the self-contained ORB
    // pipeline; mode 3 reuses the YOLOPoint/LightGlue runtime already linked
    // for feature_tracker_type 3.
    std::string loop_closure_profile = "legacy";
    if (!fsSettings["loop_closure_profile"].empty())
        fsSettings["loop_closure_profile"] >> loop_closure_profile;
    loop_closure_profile.erase(
        std::remove_if(loop_closure_profile.begin(), loop_closure_profile.end(), ::isspace),
        loop_closure_profile.end());
    std::transform(loop_closure_profile.begin(), loop_closure_profile.end(),
                   loop_closure_profile.begin(), ::tolower);
    if (loop_closure_profile != "legacy" &&
        loop_closure_profile != "indoor" &&
        loop_closure_profile != "outdoor")
    {
        ROS_WARN("loop_closure_profile must be indoor or outdoor, got '%s'; using indoor",
                 loop_closure_profile.c_str());
        loop_closure_profile = "indoor";
    }
    const bool loop_profile_indoor = loop_closure_profile == "indoor";
    const bool loop_profile_outdoor = loop_closure_profile == "outdoor";

    LOOP_CLOSURE_ENABLE = readBoolParam(fsSettings["loop_closure_enable"], 0);
    LOOP_CLOSURE_MODE = static_cast<int>(readDoubleParam(fsSettings["loop_closure_mode"], 1.0));
    if (LOOP_CLOSURE_MODE != 1 && LOOP_CLOSURE_MODE != 3)
    {
        ROS_WARN("loop_closure_mode must be 1 (ORB) or 3 (YOLOPoint), got %d; using 1",
                 LOOP_CLOSURE_MODE);
        LOOP_CLOSURE_MODE = 1;
    }
    LOOP_CLOSURE_MAX_FEATURES = std::max(100, static_cast<int>(readDoubleParam(
        fsSettings["loop_closure_max_features"], loop_profile_outdoor ? 1000.0 : 800.0)));
    LOOP_CLOSURE_KEYFRAME_STRIDE = std::max(1, static_cast<int>(readDoubleParam(fsSettings["loop_closure_keyframe_stride"], 5.0)));
    LOOP_CLOSURE_STM_SIZE = std::max(1, static_cast<int>(readDoubleParam(fsSettings["loop_closure_stm_size"], 10.0)));
    LOOP_CLOSURE_MAX_DATABASE_SIZE = std::max(0, static_cast<int>(readDoubleParam(
        fsSettings["loop_closure_max_database_size"], loop_profile_outdoor ? 3000.0 : 2000.0)));
    LOOP_CLOSURE_TEMPORAL_CONSISTENCY = std::max(1, static_cast<int>(readDoubleParam(fsSettings["loop_closure_temporal_consistency"], 2.0)));
    LOOP_CLOSURE_CANDIDATE_WINDOW = std::max(0, static_cast<int>(readDoubleParam(
        fsSettings["loop_closure_candidate_window"], loop_profile_outdoor ? 5.0 : 4.0)));
    LOOP_CLOSURE_MIN_MATCHES = std::max(6, static_cast<int>(readDoubleParam(fsSettings["loop_closure_min_matches"], 30.0)));
    LOOP_CLOSURE_MIN_INLIERS = std::max(6, static_cast<int>(readDoubleParam(fsSettings["loop_closure_min_inliers"], 20.0)));
    LOOP_CLOSURE_MIN_F_INLIERS = std::max(8, static_cast<int>(readDoubleParam(fsSettings["loop_closure_min_f_inliers"], 20.0)));
    LOOP_CLOSURE_MIN_GRID_CELLS = std::max(1, static_cast<int>(readDoubleParam(fsSettings["loop_closure_min_grid_cells"], 6.0)));
    LOOP_CLOSURE_PNP_ITERATIONS = std::max(20, static_cast<int>(readDoubleParam(fsSettings["loop_closure_pnp_iterations"], 150.0)));
    LOOP_CLOSURE_POSE_GRAPH_ITERATIONS = std::max(1, static_cast<int>(readDoubleParam(fsSettings["loop_closure_pose_graph_iterations"], 30.0)));
    LOOP_CLOSURE_MAX_PENDING_KEYFRAMES = std::max(1, static_cast<int>(readDoubleParam(fsSettings["loop_closure_max_pending_keyframes"], 5.0)));
    LOOP_CLOSURE_MIN_INTERVAL = std::max(1, static_cast<int>(readDoubleParam(fsSettings["loop_closure_min_interval"], 6.0)));
    LOOP_CLOSURE_NEIGHBOR_EDGES = std::max(1, static_cast<int>(readDoubleParam(fsSettings["loop_closure_neighbor_edges"], 4.0)));
    LOOP_CLOSURE_ALLOW_STEREO_FALLBACK = readBoolParam(fsSettings["loop_closure_allow_stereo_fallback"], 1);
    LOOP_CLOSURE_APPEARANCE_THRESHOLD = std::min(1.0, std::max(0.0, readDoubleParam(fsSettings["loop_closure_appearance_threshold"], 0.08)));
    LOOP_CLOSURE_MODEL_APPEARANCE_THRESHOLD = std::min(1.0, std::max(0.0, readDoubleParam(
        fsSettings["loop_closure_model_appearance_threshold"],
        (loop_profile_indoor || loop_profile_outdoor) ? 0.95 : 0.90)));
    LOOP_CLOSURE_MATCH_RATIO = std::min(0.99, std::max(0.1, readDoubleParam(fsSettings["loop_closure_match_ratio"], 0.75)));
    LOOP_CLOSURE_MIN_INLIER_RATIO = std::min(1.0, std::max(0.0, readDoubleParam(fsSettings["loop_closure_min_inlier_ratio"], 0.35)));
    LOOP_CLOSURE_MIN_F_INLIER_RATIO = std::min(1.0, std::max(0.0, readDoubleParam(fsSettings["loop_closure_min_f_inlier_ratio"], 0.35)));
    LOOP_CLOSURE_MIN_INLIER_SPREAD = std::min(1.0, std::max(0.0, readDoubleParam(
        fsSettings["loop_closure_min_inlier_spread"], loop_profile_outdoor ? 0.10 : 0.12)));
    LOOP_CLOSURE_F_RANSAC_THRESHOLD = std::max(0.1, readDoubleParam(fsSettings["loop_closure_f_ransac_threshold"], 1.5));
    LOOP_CLOSURE_PNP_REPROJECTION_ERROR = std::max(0.1, readDoubleParam(fsSettings["loop_closure_pnp_reprojection_error"], 3.0));
    LOOP_CLOSURE_MIN_DEPTH = std::max(0.01, readDoubleParam(
        fsSettings["loop_closure_min_depth"], loop_profile_outdoor ? 0.5 : 0.2));
    LOOP_CLOSURE_MAX_DEPTH = std::max(LOOP_CLOSURE_MIN_DEPTH, readDoubleParam(
        fsSettings["loop_closure_max_depth"], loop_profile_outdoor ? 60.0 : 30.0));
    LOOP_CLOSURE_STEREO_MAX_ERROR = std::max(0.001, readDoubleParam(
        fsSettings["loop_closure_stereo_max_error"], loop_profile_outdoor ? 0.12 : 0.08));
    LOOP_CLOSURE_MAX_TRANSLATION = std::max(0.1, readDoubleParam(
        fsSettings["loop_closure_max_translation"], loop_profile_outdoor ? 40.0 : 20.0));
    LOOP_CLOSURE_MAX_ROTATION_DEG = std::min(180.0, std::max(1.0, readDoubleParam(fsSettings["loop_closure_max_rotation_deg"], 60.0)));
    LOOP_CLOSURE_ODOM_TRANSLATION_WEIGHT = std::max(0.01, readDoubleParam(fsSettings["loop_closure_odom_translation_weight"], 5.0));
    LOOP_CLOSURE_ODOM_ROTATION_WEIGHT = std::max(0.01, readDoubleParam(fsSettings["loop_closure_odom_rotation_weight"], 10.0));
    LOOP_CLOSURE_LOOP_TRANSLATION_WEIGHT = std::max(0.01, readDoubleParam(fsSettings["loop_closure_loop_translation_weight"], 8.0));
    LOOP_CLOSURE_LOOP_ROTATION_WEIGHT = std::max(0.01, readDoubleParam(fsSettings["loop_closure_loop_rotation_weight"], 12.0));
    LOOP_CLOSURE_MIN_TRANSLATION_STD = std::max(0.001, readDoubleParam(
        fsSettings["loop_closure_min_translation_std"], loop_profile_outdoor ? 0.10 : 0.05));
    LOOP_CLOSURE_MAX_TRANSLATION_STD = std::max(LOOP_CLOSURE_MIN_TRANSLATION_STD, readDoubleParam(
        fsSettings["loop_closure_max_translation_std"], loop_profile_outdoor ? 2.0 : 1.0));
    LOOP_CLOSURE_MIN_ROTATION_STD_DEG = std::max(0.1, readDoubleParam(fsSettings["loop_closure_min_rotation_std_deg"], 1.0));
    LOOP_CLOSURE_MAX_ROTATION_STD_DEG = std::max(LOOP_CLOSURE_MIN_ROTATION_STD_DEG, readDoubleParam(fsSettings["loop_closure_max_rotation_std_deg"], 10.0));
    LOOP_CLOSURE_GRAPH_MAX_TRANSLATION_ERROR = std::max(0.01, readDoubleParam(
        fsSettings["loop_closure_graph_max_translation_error"], loop_profile_outdoor ? 4.0 : 2.0));
    LOOP_CLOSURE_GRAPH_MAX_ROTATION_ERROR_DEG = std::min(180.0, std::max(0.1, readDoubleParam(fsSettings["loop_closure_graph_max_rotation_error_deg"], 20.0)));
    LOOP_CLOSURE_GRAPH_MAX_NORMALIZED_ERROR = std::max(1.0, readDoubleParam(fsSettings["loop_closure_graph_max_normalized_error"], 5.0));
    LOOP_CLOSURE_DUAL_MAX_TRANSLATION_ERROR = std::max(0.01, readDoubleParam(
        fsSettings["loop_closure_dual_max_translation_error"], loop_profile_outdoor ? 3.0 : 1.5));
    LOOP_CLOSURE_DUAL_MAX_ROTATION_ERROR_DEG = std::min(180.0, std::max(0.1, readDoubleParam(fsSettings["loop_closure_dual_max_rotation_error_deg"], 10.0)));
    LOOP_CLOSURE_GRAVITY_WEIGHT = std::max(0.0, readDoubleParam(fsSettings["loop_closure_gravity_weight"], 50.0));
    if (!fsSettings["loop_closure_similarity_threshold"].empty())
    {
        const double similarity = std::min(1.0, std::max(
            0.0, readDoubleParam(fsSettings["loop_closure_similarity_threshold"], 0.95)));
        if (LOOP_CLOSURE_MODE == 3)
            LOOP_CLOSURE_MODEL_APPEARANCE_THRESHOLD = similarity;
        else
            LOOP_CLOSURE_APPEARANCE_THRESHOLD = similarity;
    }
    ROS_INFO("independent loop closure YAML switch: %s, mode %d (%s), profile %s, features %d, stride %d, STM %d, appearance %.3f, min inliers %d",
             LOOP_CLOSURE_ENABLE ? "true" : "false",
             LOOP_CLOSURE_MODE,
             LOOP_CLOSURE_MODE == 3 ? "YOLOPoint+LightGlue" : "ORB",
             loop_closure_profile.c_str(),
             LOOP_CLOSURE_MAX_FEATURES,
             LOOP_CLOSURE_KEYFRAME_STRIDE,
             LOOP_CLOSURE_STM_SIZE,
             LOOP_CLOSURE_MODE == 3 ? LOOP_CLOSURE_MODEL_APPEARANCE_THRESHOLD : LOOP_CLOSURE_APPEARANCE_THRESHOLD,
             LOOP_CLOSURE_MIN_INLIERS);
    ROS_INFO("independent loop effective profile: database %d, candidate window %d, spread %.3f, depth [%.2f, %.2f] m, stereo error %.3f m, max translation %.2f m",
             LOOP_CLOSURE_MAX_DATABASE_SIZE, LOOP_CLOSURE_CANDIDATE_WINDOW,
             LOOP_CLOSURE_MIN_INLIER_SPREAD, LOOP_CLOSURE_MIN_DEPTH,
             LOOP_CLOSURE_MAX_DEPTH, LOOP_CLOSURE_STEREO_MAX_ERROR,
             LOOP_CLOSURE_MAX_TRANSLATION);
    ROS_INFO("independent loop effective uncertainty: translation std [%.3f, %.3f] m, graph max %.2f m, dual max %.2f m",
             LOOP_CLOSURE_MIN_TRANSLATION_STD, LOOP_CLOSURE_MAX_TRANSLATION_STD,
             LOOP_CLOSURE_GRAPH_MAX_TRANSLATION_ERROR,
             LOOP_CLOSURE_DUAL_MAX_TRANSLATION_ERROR);

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_GPU = fsSettings["use_gpu"];
    USE_GPU_ACC_FLOW = fsSettings["use_gpu_acc_flow"];
    USE_GPU_CERES = fsSettings["use_gpu_ceres"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    USE_KITTI_Z_MOTION_PRIOR = readBoolParam(fsSettings["use_kitti_z_motion_prior"], 0);
    KITTI_Z_MOTION_PRIOR_WEIGHT = readDoubleParam(fsSettings["kitti_z_motion_prior_weight"], 0.0);
    ROS_INFO("KITTI z motion prior: %s, weight %.3f",
             USE_KITTI_Z_MOTION_PRIOR ? "true" : "false",
             KITTI_Z_MOTION_PRIOR_WEIGHT);
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    if (OUTPUT_FOLDER.empty())
        OUTPUT_FOLDER = "output";
    if (!cv::utils::fs::createDirectories(OUTPUT_FOLDER))
        throw std::runtime_error(
            "failed to create output directory: " + OUTPUT_FOLDER);
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    } 
    
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 2)
    {
        throw std::runtime_error(
            "unsupported camera configuration: num_of_cam must be 2; "
            "this project supports stereo and stereo+IMU only");
    }


    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    STEREO = 1;
    std::string cam1Calib;
    fsSettings["cam1_calib"] >> cam1Calib;
    std::string cam1Path = configPath + "/" + cam1Calib;
    CAM_NAMES.push_back(cam1Path);

    cv::Mat cv_T;
    fsSettings["body_T_cam1"] >> cv_T;
    Eigen::Matrix4d T;
    cv::cv2eigen(cv_T, T);
    RIC.push_back(T.block<3, 3>(0, 0));
    TIC.push_back(T.block<3, 1>(0, 3));

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO("Unsynchronized sensors, online estimate time offset, initial td: %f", TD);
    else
        ROS_INFO("Synchronized sensors, fix time offset: %f", TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings["world_frame_id"] >> WORLD_FRAME_ID;
    WORLD_FRAME_ID.empty()? WORLD_FRAME_ID = "world" : WORLD_FRAME_ID;
    fsSettings["body_frame_id"] >> BODY_FRAME_ID;   
    BODY_FRAME_ID.empty()? BODY_FRAME_ID = "body" : BODY_FRAME_ID;
    fsSettings["camera_frame_id"] >> CAMERA_FRAME_ID;
    CAMERA_FRAME_ID.empty()? CAMERA_FRAME_ID = "camera" : CAMERA_FRAME_ID;
    
    ROS_INFO("frame_ids: world=%s body=%s camera=%s", WORLD_FRAME_ID.c_str(),
             BODY_FRAME_ID.c_str(), CAMERA_FRAME_ID.c_str());

    fsSettings.release();
}

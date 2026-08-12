#include "independent_loop_closure.h"

#include "../estimator/parameters.h"
#include "../factor/pose_local_parameterization.h"
#include "camodocal/camera_models/Camera.h"
#include "camodocal/camera_models/CameraFactory.h"

#include <ceres/ceres.h>
#include <cv_bridge/cv_bridge.h>
#include <onnxruntime_cxx_api.h>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{
Eigen::Isometry3d makePose(const Eigen::Vector3d &p, const Eigen::Matrix3d &r)
{
    Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
    t.linear() = r;
    t.translation() = p;
    return t;
}

double rotationAngle(const Eigen::Matrix3d &r)
{
    return Eigen::AngleAxisd(r).angle();
}

double yawAngle(const Eigen::Matrix3d &r)
{
    return std::atan2(r(1, 0), r(0, 0));
}

double normalizedAngle(double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

struct PoseGraphError
{
    PoseGraphError(const Eigen::Vector3d &translation,
                   const Eigen::Quaterniond &rotation,
                   double translation_weight,
                   double rotation_weight)
        : t_(translation), q_(rotation), wt_(translation_weight), wr_(rotation_weight)
    {
    }

    template <typename U>
    bool operator()(const U *const pose_i, const U *const pose_j, U *residuals) const
    {
        Eigen::Matrix<U, 3, 1> p_i(pose_i[0], pose_i[1], pose_i[2]);
        Eigen::Matrix<U, 3, 1> p_j(pose_j[0], pose_j[1], pose_j[2]);
        Eigen::Quaternion<U> q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        Eigen::Quaternion<U> q_j(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);
        const Eigen::Matrix<U, 3, 1> predicted_t = q_i.conjugate() * (p_j - p_i);
        Eigen::Quaternion<U> predicted_q = q_i.conjugate() * q_j;
        Eigen::Quaternion<U> measured_q(U(q_.w()), U(q_.x()), U(q_.y()), U(q_.z()));
        Eigen::Quaternion<U> error_q = measured_q.conjugate() * predicted_q;
        const U sign = error_q.w() < U(0) ? U(-1) : U(1);

        residuals[0] = U(wt_) * (predicted_t.x() - U(t_.x()));
        residuals[1] = U(wt_) * (predicted_t.y() - U(t_.y()));
        residuals[2] = U(wt_) * (predicted_t.z() - U(t_.z()));
        residuals[3] = U(2.0 * wr_) * sign * error_q.x();
        residuals[4] = U(2.0 * wr_) * sign * error_q.y();
        residuals[5] = U(2.0 * wr_) * sign * error_q.z();
        return true;
    }

    Eigen::Vector3d t_;
    Eigen::Quaterniond q_;
    double wt_;
    double wr_;
};

struct GravityDirectionError
{
    GravityDirectionError(const Eigen::Vector3d &direction, double weight)
        : direction_(direction), weight_(weight)
    {
    }

    template <typename U>
    bool operator()(const U *const pose, U *residuals) const
    {
        Eigen::Quaternion<U> q(pose[6], pose[3], pose[4], pose[5]);
        const Eigen::Matrix<U, 3, 1> predicted = q * Eigen::Matrix<U, 3, 1>(U(0), U(0), U(1));
        residuals[0] = U(weight_) * (predicted.x() - U(direction_.x()));
        residuals[1] = U(weight_) * (predicted.y() - U(direction_.y()));
        residuals[2] = U(weight_) * (predicted.z() - U(direction_.z()));
        return true;
    }

    Eigen::Vector3d direction_;
    double weight_;
};

// VINS with an IMU has observable roll and pitch through gravity.  Loop
// closure is therefore only allowed to correct position and yaw, matching the
// 4-DoF pose graph used by the original VINS-Fusion loop_fusion backend.
struct FourDoFPoseGraphError
{
    FourDoFPoseGraphError(const Eigen::Vector3d &translation,
                          double relative_yaw,
                          const Eigen::Matrix3d &raw_rotation_i,
                          const Eigen::Matrix3d &raw_rotation_j,
                          double translation_weight,
                          double rotation_weight)
        : t_(translation), yaw_(relative_yaw), raw_rotation_i_(raw_rotation_i),
          raw_relative_yaw_(normalizedAngle(yawAngle(raw_rotation_j) -
                                            yawAngle(raw_rotation_i))),
          wt_(translation_weight), wr_(rotation_weight)
    {
    }

    template <typename U>
    bool operator()(const U *const yaw_i, const U *const p_i,
                    const U *const yaw_j, const U *const p_j,
                    U *residuals) const
    {
        const U c = ceres::cos(yaw_i[0]);
        const U s = ceres::sin(yaw_i[0]);
        Eigen::Matrix<U, 3, 3> correction;
        correction << c, -s, U(0), s, c, U(0), U(0), U(0), U(1);
        const Eigen::Matrix<U, 3, 3> rotation_i =
            correction * raw_rotation_i_.template cast<U>();
        const Eigen::Matrix<U, 3, 1> delta(p_j[0] - p_i[0],
                                           p_j[1] - p_i[1],
                                           p_j[2] - p_i[2]);
        const Eigen::Matrix<U, 3, 1> predicted_t = rotation_i.transpose() * delta;
        residuals[0] = U(wt_) * (predicted_t.x() - U(t_.x()));
        residuals[1] = U(wt_) * (predicted_t.y() - U(t_.y()));
        residuals[2] = U(wt_) * (predicted_t.z() - U(t_.z()));
        const U yaw_error = yaw_j[0] - yaw_i[0] + U(raw_relative_yaw_) - U(yaw_);
        residuals[3] = U(wr_) * ceres::atan2(ceres::sin(yaw_error), ceres::cos(yaw_error));
        return true;
    }

    Eigen::Vector3d t_;
    double yaw_;
    Eigen::Matrix3d raw_rotation_i_;
    double raw_relative_yaw_;
    double wt_;
    double wr_;
};
} // namespace

struct IndependentLoopClosure::Impl
{
    struct Input
    {
        double stamp = 0.0;
        cv::Mat left;
        cv::Mat right;
        Eigen::Isometry3d raw_pose = Eigen::Isometry3d::Identity();
        std::vector<Eigen::Vector3d> landmarks_world;
        std::vector<cv::Point2f> landmark_pixels;
        std::vector<int> landmark_ids;
        LoopModelFeatures model_features;
    };

    struct Keyframe
    {
        int id = -1;
        double stamp = 0.0;
        cv::Mat image;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> landmark_keypoints;
        cv::Mat landmark_descriptors;
        std::vector<Eigen::Vector3d> landmarks_world;
        std::vector<std::pair<uint32_t, float>> bow;
        cv::Mat global_descriptor;
        std::vector<Eigen::Vector3d> points_body;
        std::vector<unsigned char> has_depth;
        Eigen::Isometry3d raw_pose = Eigen::Isometry3d::Identity();
        Eigen::Isometry3d optimized_pose = Eigen::Isometry3d::Identity();
    };

    struct Edge
    {
        int from = -1;
        int to = -1;
        Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
        double translation_weight = 1.0;
        double rotation_weight = 1.0;
        double translation_std = 1.0;
        double rotation_std = 1.0;
        bool loop = false;
    };

    struct VerificationResult
    {
        bool valid = false;
        bool uses_landmarks = false;
        Eigen::Isometry3d relative = Eigen::Isometry3d::Identity();
        int matches = 0;
        int inliers = 0;
        double inlier_ratio = 0.0;
        double inlier_spread = 0.0;
        double reprojection_rmse_px = std::numeric_limits<double>::infinity();
        double median_depth = 0.0;
        double translation_std = 1.0;
        double rotation_std = M_PI / 6.0;
        std::vector<cv::DMatch> inlier_matches;
    };

    struct OdometrySample
    {
        double stamp = 0.0;
        Eigen::Isometry3d raw_pose = Eigen::Isometry3d::Identity();
        Eigen::Vector3d raw_velocity = Eigen::Vector3d::Zero();
    };

    std::atomic<bool> running{false};
    std::thread worker;
    mutable std::mutex mutex;
    std::mutex database_mutex;
    std::mutex trajectory_mutex;
    std::condition_variable condition;
    std::condition_variable idle_condition;
    bool processing = false;
    std::deque<Input> queue;
    std::vector<Keyframe> keyframes;
    std::vector<Edge> edges;
    std::vector<OdometrySample> odometry_samples;
    std::unordered_map<uint32_t, int> word_document_frequency;
    Eigen::Matrix3d r_map_odom = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_map_odom = Eigen::Vector3d::Zero();
    camodocal::CameraPtr camera_left;
    camodocal::CameraPtr camera_right;
    cv::Ptr<cv::ORB> orb;
    cv::BFMatcher matcher{cv::NORM_HAMMING, false};
    Ort::Env ort_env{ORT_LOGGING_LEVEL_WARNING, "VINSLoopLightGlue"};
    std::unique_ptr<Ort::Session> model_matcher_session;
    int last_candidate = -1;
    int consistent_hits = 0;
    int accepted_loops = 0;
    int last_verification_keyframe = -1000000;
    std::atomic<uint64_t> submitted_keyframes{0};

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr edge_publisher;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr match_publisher;

    ~Impl()
    {
        stop();
    }

    void stop()
    {
        running.store(false);
        condition.notify_all();
        if (worker.joinable())
            worker.join();
    }

    void run()
    {
        while (running.load())
        {
            Input input;
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [this] { return !running.load() || !queue.empty(); });
                if (!running.load() && queue.empty())
                    return;
                input = std::move(queue.front());
                queue.pop_front();
                processing = true;
            }
            process(std::move(input));
            {
                std::lock_guard<std::mutex> lock(mutex);
                processing = false;
            }
            idle_condition.notify_all();
        }
    }

    Eigen::Vector3d ray(const camodocal::CameraPtr &camera, const cv::Point2f &point) const
    {
        Eigen::Vector3d value;
        camera->liftProjective(Eigen::Vector2d(point.x, point.y), value);
        return value.normalized();
    }

    void extract(Keyframe &keyframe,
                 const cv::Mat &right,
                 const std::vector<Eigen::Vector3d> &landmarks_world,
                 const std::vector<cv::Point2f> &landmark_pixels,
                 const std::vector<int> &landmark_ids,
                 const LoopModelFeatures &model_features)
    {
        cv::Mat gray;
        if (keyframe.image.channels() == 3)
            cv::cvtColor(keyframe.image, gray, cv::COLOR_BGR2GRAY);
        else
            gray = keyframe.image;
        keyframe.image = gray;

        if (LOOP_CLOSURE_MODE == 3)
        {
            if (!model_features.valid())
                return;
            keyframe.keypoints = model_features.keypoints;
            keyframe.descriptors = model_features.descriptors.clone();
            keyframe.global_descriptor = model_features.global_descriptor.clone();
            keyframe.points_body.resize(keyframe.keypoints.size(), Eigen::Vector3d::Zero());
            keyframe.has_depth.resize(keyframe.keypoints.size(), 0);

            std::unordered_map<int, Eigen::Vector3d> landmark_by_id;
            const size_t associated_count =
                std::min(landmarks_world.size(), landmark_ids.size());
            for (size_t index = 0; index < associated_count; ++index)
                if (landmarks_world[index].allFinite())
                    landmark_by_id[landmark_ids[index]] = landmarks_world[index];
            for (size_t index = 0; index < model_features.feature_ids.size(); ++index)
            {
                const auto found = landmark_by_id.find(model_features.feature_ids[index]);
                if (found == landmark_by_id.end())
                    continue;
                keyframe.landmark_keypoints.push_back(keyframe.keypoints[index]);
                keyframe.landmark_descriptors.push_back(keyframe.descriptors.row(index));
                keyframe.landmarks_world.push_back(found->second);
                const Eigen::Vector3d point_body =
                    keyframe.raw_pose.inverse() * found->second;
                const Eigen::Vector3d point_camera =
                    RIC[0].transpose() * (point_body - TIC[0]);
                if (point_body.allFinite() && point_camera.z() > LOOP_CLOSURE_MIN_DEPTH &&
                    point_camera.z() < LOOP_CLOSURE_MAX_DEPTH)
                {
                    keyframe.points_body[index] = point_body;
                    keyframe.has_depth[index] = 1;
                }
            }
            keyframe.landmark_descriptors = keyframe.landmark_descriptors.clone();

            std::vector<cv::Point2f> stereo_right_points =
                model_features.right_points;
            std::vector<unsigned char> stereo_status =
                model_features.has_right;
            int recovered_stereo = 0;
            const Eigen::Matrix3d r_c1_c0 = RIC[1].transpose() * RIC[0];
            const Eigen::Vector3d t_c1_c0 =
                RIC[1].transpose() * (TIC[0] - TIC[1]);
            Eigen::Matrix3d translation_skew;
            translation_skew <<
                0.0, -t_c1_c0.z(), t_c1_c0.y(),
                t_c1_c0.z(), 0.0, -t_c1_c0.x(),
                -t_c1_c0.y(), t_c1_c0.x(), 0.0;
            const Eigen::Matrix3d stereo_essential =
                translation_skew * r_c1_c0;
            if (!right.empty())
            {
                cv::Mat right_gray;
                if (right.channels() == 3)
                    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
                else
                    right_gray = right;
                cv::Mat left_corner_eigenvalues;
                if (YOLOPOINT_LIGHTGLUE_STEREO_MIN_CORNER_EIGENVALUE > 0.0)
                    cv::cornerMinEigenVal(
                        gray, left_corner_eigenvalues, 7, 3);
                std::vector<int> missing_indices;
                std::vector<cv::Point2f> missing_left_points;
                missing_indices.reserve(keyframe.keypoints.size());
                missing_left_points.reserve(keyframe.keypoints.size());
                for (size_t index = 0; index < keyframe.keypoints.size(); ++index)
                {
                    if (keyframe.has_depth[index] || stereo_status[index])
                        continue;
                    missing_indices.push_back(static_cast<int>(index));
                    missing_left_points.push_back(keyframe.keypoints[index].pt);
                }
                if (!missing_left_points.empty())
                {
                    std::vector<cv::Point2f> tracked_right_points;
                    std::vector<unsigned char> forward_status;
                    std::vector<float> errors;
                    cv::calcOpticalFlowPyrLK(
                        gray, right_gray, missing_left_points,
                        tracked_right_points, forward_status, errors,
                        cv::Size(21, 21), 3);
                    std::vector<cv::Point2f> reverse_left_points;
                    std::vector<unsigned char> reverse_status;
                    if (FLOW_BACK)
                    {
                        cv::calcOpticalFlowPyrLK(
                            right_gray, gray, tracked_right_points,
                            reverse_left_points, reverse_status, errors,
                            cv::Size(21, 21), 3);
                    }
                    for (size_t local = 0; local < missing_indices.size(); ++local)
                    {
                        if (!forward_status[local])
                            continue;
                        const cv::Point2f &right_point = tracked_right_points[local];
                        if (right_point.x < 1.0F || right_point.y < 1.0F ||
                            right_point.x >= static_cast<float>(right_gray.cols - 1) ||
                            right_point.y >= static_cast<float>(right_gray.rows - 1))
                            continue;
                        if (FLOW_BACK &&
                            (local >= reverse_status.size() || !reverse_status[local] ||
                             cv::norm(missing_left_points[local] -
                                      reverse_left_points[local]) > 0.5))
                            continue;
                        const int index = missing_indices[local];
                        if (!left_corner_eigenvalues.empty())
                        {
                            const cv::Point point(
                                cvRound(missing_left_points[local].x),
                                cvRound(missing_left_points[local].y));
                            if (point.x < 0 || point.y < 0 ||
                                point.x >= left_corner_eigenvalues.cols ||
                                point.y >= left_corner_eigenvalues.rows ||
                                left_corner_eigenvalues.at<float>(point) <
                                    YOLOPOINT_LIGHTGLUE_STEREO_MIN_CORNER_EIGENVALUE)
                                continue;
                        }
                        Eigen::Vector3d left_ray =
                            ray(camera_left, missing_left_points[local]);
                        Eigen::Vector3d right_ray =
                            ray(camera_right, right_point);
                        if (std::abs(left_ray.z()) <= 1e-12 ||
                            std::abs(right_ray.z()) <= 1e-12)
                            continue;
                        left_ray /= left_ray.z();
                        right_ray /= right_ray.z();
                        const Eigen::Vector3d epipolar_line =
                            stereo_essential * left_ray;
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
                        Eigen::Matrix<double, 3, 2> ray_system;
                        ray_system.col(0) = r_c1_c0 * left_ray;
                        ray_system.col(1) = -right_ray;
                        const Eigen::Vector2d ray_depths =
                            ray_system.colPivHouseholderQr().solve(-t_c1_c0);
                        const Eigen::Vector3d point1_from_left =
                            r_c1_c0 * (ray_depths.x() * left_ray) + t_c1_c0;
                        const Eigen::Vector3d point1_from_right =
                            ray_depths.y() * right_ray;
                        const Eigen::Vector3d point1 =
                            0.5 * (point1_from_left + point1_from_right);
                        const Eigen::Vector3d point0 =
                            r_c1_c0.transpose() * (point1 - t_c1_c0);
                        if (!point0.allFinite() || !point1.allFinite() ||
                            point0.z() < YOLOPOINT_LIGHTGLUE_STEREO_MIN_DEPTH ||
                            point1.z() < YOLOPOINT_LIGHTGLUE_STEREO_MIN_DEPTH ||
                            point0.z() > YOLOPOINT_LIGHTGLUE_STEREO_MAX_DEPTH ||
                            point1.z() > YOLOPOINT_LIGHTGLUE_STEREO_MAX_DEPTH)
                            continue;
                        const Eigen::Vector2d left_reprojection =
                            point0.head<2>() / point0.z();
                        const Eigen::Vector2d right_reprojection =
                            point1.head<2>() / point1.z();
                        const double left_reprojection_error =
                            (left_reprojection - left_ray.head<2>()).norm() *
                            FOCAL_LENGTH;
                        const double right_reprojection_error =
                            (right_reprojection - right_ray.head<2>()).norm() *
                            FOCAL_LENGTH;
                        if (!std::isfinite(left_reprojection_error) ||
                            !std::isfinite(right_reprojection_error) ||
                            std::max(left_reprojection_error,
                                     right_reprojection_error) >
                                YOLOPOINT_LIGHTGLUE_STEREO_REPROJECTION_THRESHOLD)
                            continue;
                        stereo_right_points[index] = right_point;
                        stereo_status[index] = 1;
                        ++recovered_stereo;
                    }
                }
            }

            for (size_t index = 0; index < keyframe.keypoints.size(); ++index)
            {
                if (keyframe.has_depth[index])
                    continue;
                if (!stereo_status[index])
                    continue;
                const Eigen::Vector3d left_ray =
                    ray(camera_left, keyframe.keypoints[index].pt);
                const Eigen::Vector3d right_ray =
                    ray(camera_right, stereo_right_points[index]);
                Eigen::Matrix<double, 3, 2> a;
                a.col(0) = r_c1_c0 * left_ray;
                a.col(1) = -right_ray;
                const Eigen::Vector2d depths =
                    a.colPivHouseholderQr().solve(-t_c1_c0);
                const Eigen::Vector3d residual = a * depths + t_c1_c0;
                if (depths.x() <= LOOP_CLOSURE_MIN_DEPTH ||
                    depths.y() <= LOOP_CLOSURE_MIN_DEPTH ||
                    depths.x() >= LOOP_CLOSURE_MAX_DEPTH ||
                    residual.norm() > LOOP_CLOSURE_STEREO_MAX_ERROR)
                    continue;
                const Eigen::Vector3d point_camera = depths.x() * left_ray;
                keyframe.points_body[index] = RIC[0] * point_camera + TIC[0];
                keyframe.has_depth[index] = 1;
            }
            if (keyframe.id % 50 == 0)
            {
                const int depth_count = std::count(
                    keyframe.has_depth.begin(), keyframe.has_depth.end(),
                    static_cast<unsigned char>(1));
                ROS_INFO("model loop keyframe %d: full features %zu, depth %d, recovered stereo tracks %d",
                         keyframe.id, keyframe.keypoints.size(), depth_count,
                         recovered_stereo);
            }
            return;
        }

        orb->detectAndCompute(gray, cv::noArray(), keyframe.keypoints, keyframe.descriptors);
        if (!keyframe.descriptors.empty())
        {
            std::vector<uint32_t> words;
            constexpr int table_count = 8;
            constexpr int bits_per_word = 14;
            words.reserve(static_cast<size_t>(keyframe.descriptors.rows) * table_count);
            for (int row = 0; row < keyframe.descriptors.rows; ++row)
            {
                const unsigned char *descriptor = keyframe.descriptors.ptr<unsigned char>(row);
                // Multi-table binary locality-sensitive words use bits spread
                // over the complete ORB descriptor.  Unlike the old two-byte
                // truncation, similar descriptors can share one or more words
                // without requiring an external trained vocabulary.
                for (int table = 0; table < table_count; ++table)
                {
                    uint32_t code = 0;
                    for (int bit = 0; bit < bits_per_word; ++bit)
                    {
                        const int descriptor_bit =
                            (table * 37 + bit * 53 + table * bit * 7) & 255;
                        const uint32_t value =
                            (descriptor[descriptor_bit / 8] >> (descriptor_bit % 8)) & 1U;
                        code |= value << bit;
                    }
                    const uint32_t word = (static_cast<uint32_t>(table) << bits_per_word) | code;
                    words.push_back(word);
                }
            }
            std::sort(words.begin(), words.end());
            const float normalization = 1.0F / static_cast<float>(words.size());
            for (size_t begin = 0; begin < words.size();)
            {
                size_t end = begin + 1;
                while (end < words.size() && words[end] == words[begin])
                    ++end;
                keyframe.bow.emplace_back(words[begin],
                                          static_cast<float>(end - begin) * normalization);
                begin = end;
            }
        }

        const size_t landmark_count = std::min(landmarks_world.size(), landmark_pixels.size());
        keyframe.landmark_keypoints.reserve(landmark_count);
        for (size_t i = 0; i < landmark_count; ++i)
        {
            const cv::Point2f &pixel = landmark_pixels[i];
            if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) ||
                pixel.x < 16.0F || pixel.y < 16.0F ||
                pixel.x >= static_cast<float>(gray.cols - 16) ||
                pixel.y >= static_cast<float>(gray.rows - 16) ||
                !landmarks_world[i].allFinite())
                continue;
            cv::KeyPoint keypoint(pixel, 31.0F);
            keypoint.class_id = static_cast<int>(i);
            keyframe.landmark_keypoints.push_back(keypoint);
        }
        if (!keyframe.landmark_keypoints.empty())
        {
            orb->compute(gray, keyframe.landmark_keypoints, keyframe.landmark_descriptors);
            keyframe.landmarks_world.reserve(keyframe.landmark_keypoints.size());
            for (const cv::KeyPoint &keypoint : keyframe.landmark_keypoints)
            {
                if (keypoint.class_id >= 0 && keypoint.class_id < static_cast<int>(landmarks_world.size()))
                    keyframe.landmarks_world.push_back(landmarks_world[keypoint.class_id]);
            }
            if (keyframe.landmarks_world.size() != keyframe.landmark_keypoints.size())
            {
                keyframe.landmark_keypoints.clear();
                keyframe.landmark_descriptors.release();
                keyframe.landmarks_world.clear();
            }
        }
        keyframe.points_body.resize(keyframe.keypoints.size(), Eigen::Vector3d::Zero());
        keyframe.has_depth.resize(keyframe.keypoints.size(), 0);

        if (right.empty() || keyframe.descriptors.empty())
            return;

        cv::Mat right_gray;
        if (right.channels() == 3)
            cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
        else
            right_gray = right;
        std::vector<cv::KeyPoint> right_keypoints;
        cv::Mat right_descriptors;
        orb->detectAndCompute(right_gray, cv::noArray(), right_keypoints, right_descriptors);
        if (right_descriptors.empty())
            return;

        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(keyframe.descriptors, right_descriptors, matches, 2);
        const Eigen::Matrix3d r_c1_c0 = RIC[1].transpose() * RIC[0];
        const Eigen::Vector3d t_c1_c0 = RIC[1].transpose() * (TIC[0] - TIC[1]);
        for (size_t i = 0; i < matches.size(); ++i)
        {
            if (matches[i].size() < 2 || matches[i][0].distance >= LOOP_CLOSURE_MATCH_RATIO * matches[i][1].distance)
                continue;
            const int right_index = matches[i][0].trainIdx;
            Eigen::Vector3d left_ray = ray(camera_left, keyframe.keypoints[i].pt);
            Eigen::Vector3d right_ray = ray(camera_right, right_keypoints[right_index].pt);
            Eigen::Matrix<double, 3, 2> a;
            a.col(0) = r_c1_c0 * left_ray;
            a.col(1) = -right_ray;
            Eigen::Vector2d depths = a.colPivHouseholderQr().solve(-t_c1_c0);
            const Eigen::Vector3d residual = a * depths + t_c1_c0;
            if (depths.x() <= LOOP_CLOSURE_MIN_DEPTH || depths.y() <= LOOP_CLOSURE_MIN_DEPTH ||
                depths.x() >= LOOP_CLOSURE_MAX_DEPTH || residual.norm() > LOOP_CLOSURE_STEREO_MAX_ERROR)
                continue;
            const Eigen::Vector3d point_camera = depths.x() * left_ray;
            keyframe.points_body[i] = RIC[0] * point_camera + TIC[0];
            keyframe.has_depth[i] = 1;
        }
    }

    void initializeModelMatcher()
    {
        auto make_options = [](bool use_cuda)
        {
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            options.SetIntraOpNumThreads(1);
            if (use_cuda)
            {
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = 0;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
                cuda_options.gpu_mem_limit = 0;
                cuda_options.arena_extend_strategy = 1;
                cuda_options.do_copy_in_default_stream = 1;
                options.AppendExecutionProvider_CUDA(cuda_options);
            }
            return options;
        };

        if (YOLOPOINT_LIGHTGLUE_USE_CUDA)
        {
            try
            {
                Ort::SessionOptions options = make_options(true);
                model_matcher_session.reset(new Ort::Session(
                    ort_env, YOLOPOINT_LIGHTGLUE_MATCHER_MODEL_PATH.c_str(), options));
                return;
            }
            catch (const std::exception &error)
            {
                ROS_WARN("loop LightGlue CUDA session failed: %s; falling back to CPU",
                         error.what());
            }
        }
        Ort::SessionOptions options = make_options(false);
        model_matcher_session.reset(new Ort::Session(
            ort_env, YOLOPOINT_LIGHTGLUE_MATCHER_MODEL_PATH.c_str(), options));
    }

    std::vector<cv::DMatch> modelMatches(
        const std::vector<cv::KeyPoint> &query_keypoints,
        const cv::Mat &query_descriptors,
        const cv::Size &query_size,
        const std::vector<cv::KeyPoint> &train_keypoints,
        const cv::Mat &train_descriptors,
        const cv::Size &train_size) const
    {
        std::vector<cv::DMatch> accepted;
        if (!model_matcher_session || query_keypoints.empty() ||
            train_keypoints.empty() || query_descriptors.empty() ||
            train_descriptors.empty())
            return accepted;
        const int query_count = std::min(
            static_cast<int>(query_keypoints.size()), query_descriptors.rows);
        const int train_count = std::min(
            static_cast<int>(train_keypoints.size()), train_descriptors.rows);
        if (query_count <= 0 || train_count <= 0 ||
            query_descriptors.cols != 128 || train_descriptors.cols != 128)
            return accepted;

        std::vector<float> query_points(query_count * 2);
        std::vector<float> train_points(train_count * 2);
        for (int index = 0; index < query_count; ++index)
        {
            query_points[2 * index] = query_keypoints[index].pt.x;
            query_points[2 * index + 1] = query_keypoints[index].pt.y;
        }
        for (int index = 0; index < train_count; ++index)
        {
            train_points[2 * index] = train_keypoints[index].pt.x;
            train_points[2 * index + 1] = train_keypoints[index].pt.y;
        }
        cv::Mat query_contiguous = query_descriptors.isContinuous()
                                       ? query_descriptors
                                       : query_descriptors.clone();
        cv::Mat train_contiguous = train_descriptors.isContinuous()
                                       ? train_descriptors
                                       : train_descriptors.clone();
        std::vector<float> query_image_size{
            static_cast<float>(query_size.width),
            static_cast<float>(query_size.height)};
        std::vector<float> train_image_size{
            static_cast<float>(train_size.width),
            static_cast<float>(train_size.height)};
        const std::vector<int64_t> query_keypoint_shape{1, query_count, 2};
        const std::vector<int64_t> train_keypoint_shape{1, train_count, 2};
        const std::vector<int64_t> query_descriptor_shape{1, query_count, 128};
        const std::vector<int64_t> train_descriptor_shape{1, train_count, 128};
        const std::vector<int64_t> image_size_shape{1, 2};
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, query_points.data(), query_points.size(),
            query_keypoint_shape.data(), query_keypoint_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, train_points.data(), train_points.size(),
            train_keypoint_shape.data(), train_keypoint_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, query_contiguous.ptr<float>(),
            static_cast<size_t>(query_count) * 128,
            query_descriptor_shape.data(), query_descriptor_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, train_contiguous.ptr<float>(),
            static_cast<size_t>(train_count) * 128,
            train_descriptor_shape.data(), train_descriptor_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, query_image_size.data(), query_image_size.size(),
            image_size_shape.data(), image_size_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, train_image_size.data(), train_image_size.size(),
            image_size_shape.data(), image_size_shape.size()));
        const char *input_names[] = {
            "keypoints0", "keypoints1", "descriptors0",
            "descriptors1", "image_size0", "image_size1"};
        const char *output_names[] = {"matches0", "matching_scores0"};
        auto outputs = model_matcher_session->Run(
            Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(),
            output_names, 2);
        const int64_t *matches = outputs[0].GetTensorData<int64_t>();
        const float *scores = outputs[1].GetTensorData<float>();
        accepted.reserve(query_count);
        for (int query = 0; query < query_count; ++query)
        {
            const int64_t train = matches[query];
            if (train >= 0 && train < train_count)
                accepted.emplace_back(query, static_cast<int>(train),
                                      1.0f - scores[query]);
        }
        return accepted;
    }

    double appearanceScore(const Keyframe &current, const Keyframe &candidate) const
    {
        if (LOOP_CLOSURE_MODE == 3)
        {
            if (current.global_descriptor.empty() ||
                candidate.global_descriptor.empty() ||
                current.global_descriptor.cols != candidate.global_descriptor.cols)
                return 0.0;
            return std::max(-1.0, std::min(1.0,
                current.global_descriptor.dot(candidate.global_descriptor)));
        }
        if (current.bow.empty() || candidate.bow.empty())
            return 0.0;
        size_t current_index = 0;
        size_t candidate_index = 0;
        double intersection = 0.0;
        double current_norm = 0.0;
        double candidate_norm = 0.0;
        const double document_count = static_cast<double>(keyframes.size()) + 1.0;
        for (const auto &word : current.bow)
        {
            const auto found = word_document_frequency.find(word.first);
            const double frequency = found == word_document_frequency.end() ? 0.0 : found->second;
            current_norm += word.second * (std::log(document_count / (frequency + 1.0)) + 1.0);
        }
        for (const auto &word : candidate.bow)
        {
            const auto found = word_document_frequency.find(word.first);
            const double frequency = found == word_document_frequency.end() ? 0.0 : found->second;
            candidate_norm += word.second * (std::log(document_count / (frequency + 1.0)) + 1.0);
        }
        while (current_index < current.bow.size() && candidate_index < candidate.bow.size())
        {
            if (current.bow[current_index].first < candidate.bow[candidate_index].first)
                ++current_index;
            else if (candidate.bow[candidate_index].first < current.bow[current_index].first)
                ++candidate_index;
            else
            {
                const auto found = word_document_frequency.find(current.bow[current_index].first);
                const double frequency = found == word_document_frequency.end() ? 0.0 : found->second;
                const double idf = std::log(document_count / (frequency + 1.0)) + 1.0;
                intersection += idf * std::min(current.bow[current_index].second,
                                               candidate.bow[candidate_index].second);
                ++current_index;
                ++candidate_index;
            }
        }
        if (current_norm <= 0.0 || candidate_norm <= 0.0)
            return 0.0;
        // Online TF-IDF keeps retrieval self-contained while suppressing words
        // occurring in nearly every scene. Multi-table word overlap already
        // has a natural [0,1] Dice normalization, so no score inflation is used.
        return std::min(1.0, 2.0 * intersection / (current_norm + candidate_norm));
    }

    void addDocument(const Keyframe &keyframe)
    {
        if (LOOP_CLOSURE_MODE == 3)
            return;
        for (const auto &word : keyframe.bow)
            ++word_document_frequency[word.first];
    }

    int detectCandidate(const Keyframe &current, double &score)
    {
        score = 0.0;
        int best = -1;
        const int searchable = static_cast<int>(keyframes.size()) - LOOP_CLOSURE_STM_SIZE;
        if (searchable <= 0)
            return -1;
        const int first = LOOP_CLOSURE_MAX_DATABASE_SIZE > 0
                              ? std::max(0, searchable - LOOP_CLOSURE_MAX_DATABASE_SIZE)
                              : 0;
        for (int i = first; i < searchable; ++i)
        {
            const double raw = appearanceScore(current, keyframes[i]);
            const double temporal_prior = last_candidate >= 0
                                              ? std::exp(-std::abs(i - last_candidate) / 3.0)
                                              : 0.0;
            const double posterior = last_candidate >= 0 ? 0.75 * raw + 0.25 * raw * temporal_prior : raw;
            if (posterior > score)
            {
                score = posterior;
                best = i;
            }
        }
        const double threshold = LOOP_CLOSURE_MODE == 3
                                     ? LOOP_CLOSURE_MODEL_APPEARANCE_THRESHOLD
                                     : LOOP_CLOSURE_APPEARANCE_THRESHOLD;
        if (score < threshold)
        {
            last_candidate = -1;
            consistent_hits = 0;
            return -1;
        }
        if (last_candidate >= 0 && std::abs(best - last_candidate) <= LOOP_CLOSURE_CANDIDATE_WINDOW)
            ++consistent_hits;
        else
            consistent_hits = 1;
        last_candidate = best;
        return consistent_hits >= LOOP_CLOSURE_TEMPORAL_CONSISTENCY ? best : -1;
    }

    double inlierSpread(const std::vector<cv::Point2f> &points,
                        const cv::Mat &inliers,
                        int width,
                        int height) const
    {
        if (inliers.rows <= 0 || width <= 0 || height <= 0)
            return 0.0;
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        for (int i = 0; i < inliers.rows; ++i)
        {
            const cv::Point2f &point = points[inliers.at<int>(i)];
            min_x = std::min(min_x, point.x);
            min_y = std::min(min_y, point.y);
            max_x = std::max(max_x, point.x);
            max_y = std::max(max_y, point.y);
        }
        const double diagonal = std::hypot((max_x - min_x) / width, (max_y - min_y) / height);
        return std::min(1.0, diagonal / std::sqrt(2.0));
    }

    std::vector<cv::DMatch> mutualRatioMatches(const cv::Mat &query,
                                                const cv::Mat &train) const
    {
        std::vector<cv::DMatch> accepted;
        if (query.empty() || train.empty())
            return accepted;
        std::vector<std::vector<cv::DMatch>> forward_pairs;
        std::vector<cv::DMatch> reverse_matches;
        matcher.knnMatch(query, train, forward_pairs, 2);
        matcher.match(train, query, reverse_matches);
        std::vector<int> reverse_best(train.rows, -1);
        for (const cv::DMatch &match : reverse_matches)
            if (match.queryIdx >= 0 && match.queryIdx < train.rows)
                reverse_best[match.queryIdx] = match.trainIdx;
        for (const auto &pair : forward_pairs)
        {
            if (pair.size() < 2 ||
                pair[0].distance >= LOOP_CLOSURE_MATCH_RATIO * pair[1].distance)
                continue;
            const cv::DMatch &match = pair[0];
            if (match.trainIdx >= 0 && match.trainIdx < train.rows &&
                reverse_best[match.trainIdx] == match.queryIdx)
                accepted.push_back(match);
        }
        return accepted;
    }

    bool fundamentalInlierMask(const std::vector<cv::Point2f> &points_a,
                               const std::vector<cv::Point2f> &points_b,
                               std::vector<unsigned char> &mask) const
    {
        mask.assign(points_a.size(), 0);
        if (points_a.size() < 8 || points_a.size() != points_b.size())
            return false;
        std::vector<cv::Point2f> normalized_a;
        std::vector<cv::Point2f> normalized_b;
        normalized_a.reserve(points_a.size());
        normalized_b.reserve(points_b.size());
        for (size_t i = 0; i < points_a.size(); ++i)
        {
            const Eigen::Vector3d ray_a = ray(camera_left, points_a[i]);
            const Eigen::Vector3d ray_b = ray(camera_left, points_b[i]);
            if (std::abs(ray_a.z()) < 1e-9 || std::abs(ray_b.z()) < 1e-9)
                return false;
            normalized_a.emplace_back(static_cast<float>(FOCAL_LENGTH * ray_a.x() / ray_a.z()),
                                      static_cast<float>(FOCAL_LENGTH * ray_a.y() / ray_a.z()));
            normalized_b.emplace_back(static_cast<float>(FOCAL_LENGTH * ray_b.x() / ray_b.z()),
                                      static_cast<float>(FOCAL_LENGTH * ray_b.y() / ray_b.z()));
        }
        const cv::Mat fundamental = cv::findFundamentalMat(
            normalized_a, normalized_b, cv::FM_RANSAC,
            LOOP_CLOSURE_F_RANSAC_THRESHOLD, 0.99, mask);
        const int inliers = cv::countNonZero(mask);
        return !fundamental.empty() &&
               inliers >= LOOP_CLOSURE_MIN_F_INLIERS &&
               static_cast<double>(inliers) / points_a.size() >=
                   LOOP_CLOSURE_MIN_F_INLIER_RATIO;
    }

    int occupiedGridCells(const std::vector<cv::Point2f> &points,
                          const cv::Mat &inliers,
                          int width,
                          int height) const
    {
        constexpr int rows = 4;
        constexpr int cols = 4;
        std::array<unsigned char, rows * cols> occupied{};
        for (int i = 0; i < inliers.rows; ++i)
        {
            const int index = inliers.at<int>(i);
            if (index < 0 || index >= static_cast<int>(points.size()))
                continue;
            const int x = std::min(cols - 1, std::max(0,
                static_cast<int>(points[index].x * cols / std::max(1, width))));
            const int y = std::min(rows - 1, std::max(0,
                static_cast<int>(points[index].y * rows / std::max(1, height))));
            occupied[y * cols + x] = 1;
        }
        return std::count(occupied.begin(), occupied.end(), static_cast<unsigned char>(1));
    }

    void estimatePnPUncertainty(const std::vector<cv::Point3f> &object_points,
                                const std::vector<cv::Point2f> &image_points,
                                const cv::Mat &rvec,
                                const cv::Mat &tvec,
                                const cv::Mat &inliers,
                                VerificationResult &result) const
    {
        if (inliers.rows < 6)
            return;

        std::vector<cv::Point3f> inlier_objects;
        std::vector<cv::Point2f> inlier_images;
        inlier_objects.reserve(inliers.rows);
        inlier_images.reserve(inliers.rows);
        for (int row = 0; row < inliers.rows; ++row)
        {
            const int index = inliers.at<int>(row);
            if (index < 0 || index >= static_cast<int>(object_points.size()))
                continue;
            inlier_objects.push_back(object_points[index]);
            inlier_images.push_back(image_points[index]);
        }
        if (inlier_objects.size() < 6)
            return;

        cv::Mat jacobian;
        std::vector<cv::Point2f> projections;
        cv::projectPoints(inlier_objects, rvec, tvec, cv::Mat::eye(3, 3, CV_64F),
                          cv::Mat(), projections, jacobian);
        double squared_error = 0.0;
        for (size_t i = 0; i < projections.size(); ++i)
        {
            const cv::Point2f error = projections[i] - inlier_images[i];
            squared_error += error.dot(error);
        }
        const double normalized_rmse = std::sqrt(
            squared_error / std::max<size_t>(1, projections.size()));
        result.reprojection_rmse_px = normalized_rmse * FOCAL_LENGTH;

        cv::Mat rotation_cv;
        cv::Rodrigues(rvec, rotation_cv);
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        cv::cv2eigen(rotation_cv, rotation);
        cv::cv2eigen(tvec, translation);
        std::vector<double> depths;
        depths.reserve(inlier_objects.size());
        for (const cv::Point3f &point : inlier_objects)
        {
            const Eigen::Vector3d camera_point =
                rotation * Eigen::Vector3d(point.x, point.y, point.z) + translation;
            if (camera_point.z() > 0.0 && std::isfinite(camera_point.z()))
                depths.push_back(camera_point.z());
        }
        if (!depths.empty())
        {
            const size_t middle = depths.size() / 2;
            std::nth_element(depths.begin(), depths.begin() + middle, depths.end());
            result.median_depth = depths[middle];
        }

        double covariance_translation_std = LOOP_CLOSURE_MAX_TRANSLATION_STD;
        double covariance_rotation_std = LOOP_CLOSURE_MAX_ROTATION_STD_DEG * M_PI / 180.0;
        if (jacobian.rows >= static_cast<int>(2 * inlier_objects.size()) && jacobian.cols >= 6)
        {
            const cv::Mat pose_jacobian = jacobian.colRange(0, 6);
            cv::Mat normal = pose_jacobian.t() * pose_jacobian;
            cv::Mat inverse_normal;
            if (cv::invert(normal, inverse_normal, cv::DECOMP_SVD))
            {
                const double pixel_floor = 0.5 / std::max(1.0, FOCAL_LENGTH);
                const double sigma = std::max(pixel_floor, normalized_rmse);
                const cv::Mat covariance = inverse_normal * (sigma * sigma);
                covariance_rotation_std = std::sqrt(std::max({
                    covariance.at<double>(0, 0), covariance.at<double>(1, 1),
                    covariance.at<double>(2, 2), 0.0}));
                covariance_translation_std = std::sqrt(std::max({
                    covariance.at<double>(3, 3), covariance.at<double>(4, 4),
                    covariance.at<double>(5, 5), 0.0}));
            }
        }

        // Reprojection covariance alone is over-confident for stereo points:
        // disparity uncertainty grows quadratically with depth.  Add a small
        // stereo-depth floor so far, low-parallax loops cannot receive the
        // same graph weight as nearby well-conditioned loops.
        const double baseline = (TIC[1] - TIC[0]).norm();
        double stereo_depth_std = 0.0;
        if (baseline > 1e-4 && result.median_depth > 0.0)
            stereo_depth_std = 0.25 * result.median_depth * result.median_depth /
                               (std::max(1.0, FOCAL_LENGTH) * baseline);
        result.translation_std = std::min(LOOP_CLOSURE_MAX_TRANSLATION_STD,
            std::max(LOOP_CLOSURE_MIN_TRANSLATION_STD,
                     std::hypot(covariance_translation_std, stereo_depth_std)));
        result.rotation_std = std::min(LOOP_CLOSURE_MAX_ROTATION_STD_DEG * M_PI / 180.0,
            std::max(LOOP_CLOSURE_MIN_ROTATION_STD_DEG * M_PI / 180.0,
                     covariance_rotation_std));
    }

    VerificationResult stereoGeometricVerification(const Keyframe &candidate,
                                                    const Keyframe &current)
    {
        VerificationResult result;
        if (candidate.descriptors.empty() || current.descriptors.empty())
            return result;
        const std::vector<cv::DMatch> matches = LOOP_CLOSURE_MODE == 3
            ? modelMatches(candidate.keypoints, candidate.descriptors,
                           candidate.image.size(), current.keypoints,
                           current.descriptors, current.image.size())
            : mutualRatioMatches(candidate.descriptors, current.descriptors);
        std::vector<cv::Point3f> object_points;
        std::vector<cv::Point2f> image_points;
        std::vector<cv::Point2f> candidate_pixels;
        std::vector<cv::Point2f> current_pixels;
        std::vector<cv::DMatch> used_matches;
        for (const cv::DMatch &match : matches)
        {
            const int query = match.queryIdx;
            if (query < 0 || query >= static_cast<int>(candidate.has_depth.size()) || !candidate.has_depth[query])
                continue;
            const Eigen::Vector3d point = candidate.points_body[query];
            const Eigen::Vector3d current_ray = ray(camera_left, current.keypoints[match.trainIdx].pt);
            if (std::abs(current_ray.z()) < 1e-6)
                continue;
            object_points.emplace_back(point.x(), point.y(), point.z());
            image_points.emplace_back(current_ray.x() / current_ray.z(), current_ray.y() / current_ray.z());
            candidate_pixels.push_back(candidate.keypoints[match.queryIdx].pt);
            current_pixels.push_back(current.keypoints[match.trainIdx].pt);
            used_matches.push_back(match);
        }
        if (static_cast<int>(object_points.size()) < LOOP_CLOSURE_MIN_MATCHES)
            return result;

        std::vector<unsigned char> f_mask;
        if (!fundamentalInlierMask(candidate_pixels, current_pixels, f_mask))
            return result;
        for (int i = static_cast<int>(f_mask.size()) - 1; i >= 0; --i)
        {
            if (f_mask[i])
                continue;
            object_points.erase(object_points.begin() + i);
            image_points.erase(image_points.begin() + i);
            candidate_pixels.erase(candidate_pixels.begin() + i);
            current_pixels.erase(current_pixels.begin() + i);
            used_matches.erase(used_matches.begin() + i);
        }
        if (static_cast<int>(object_points.size()) < LOOP_CLOSURE_MIN_MATCHES)
            return result;

        cv::Mat rvec, tvec, inliers;
        const cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        const double normalized_threshold = LOOP_CLOSURE_PNP_REPROJECTION_ERROR / FOCAL_LENGTH;
        const bool solved = cv::solvePnPRansac(object_points,
                                               image_points,
                                               camera_matrix,
                                               cv::Mat(),
                                               rvec,
                                               tvec,
                                               false,
                                               LOOP_CLOSURE_PNP_ITERATIONS,
                                               normalized_threshold,
                                               0.995,
                                               inliers,
                                               cv::SOLVEPNP_EPNP);
        if (!solved || inliers.rows < LOOP_CLOSURE_MIN_INLIERS)
            return result;

        cv::Mat rotation_cv;
        cv::Rodrigues(rvec, rotation_cv);
        Eigen::Matrix3d r_camera_current_body_candidate;
        Eigen::Vector3d t_camera_current_body_candidate;
        cv::cv2eigen(rotation_cv, r_camera_current_body_candidate);
        cv::cv2eigen(tvec, t_camera_current_body_candidate);
        Eigen::Isometry3d t_camera_current_body_candidate_iso = Eigen::Isometry3d::Identity();
        t_camera_current_body_candidate_iso.linear() = r_camera_current_body_candidate;
        t_camera_current_body_candidate_iso.translation() = t_camera_current_body_candidate;
        Eigen::Isometry3d t_body_current_camera_current = Eigen::Isometry3d::Identity();
        t_body_current_camera_current.linear() = RIC[0];
        t_body_current_camera_current.translation() = TIC[0];
        result.relative = (t_body_current_camera_current * t_camera_current_body_candidate_iso).inverse();

        result.matches = static_cast<int>(object_points.size());
        result.inliers = inliers.rows;
        result.inlier_ratio = static_cast<double>(result.inliers) / result.matches;
        result.inlier_spread = inlierSpread(current_pixels, inliers,
                                            current.image.cols, current.image.rows);
        estimatePnPUncertainty(object_points, image_points, rvec, tvec, inliers, result);

        if (result.relative.translation().norm() > LOOP_CLOSURE_MAX_TRANSLATION ||
            rotationAngle(result.relative.linear()) > LOOP_CLOSURE_MAX_ROTATION_DEG * M_PI / 180.0 ||
            result.inlier_ratio < LOOP_CLOSURE_MIN_INLIER_RATIO ||
            result.inlier_spread < LOOP_CLOSURE_MIN_INLIER_SPREAD ||
            occupiedGridCells(candidate_pixels, inliers, candidate.image.cols,
                              candidate.image.rows) < LOOP_CLOSURE_MIN_GRID_CELLS ||
            occupiedGridCells(current_pixels, inliers, current.image.cols,
                              current.image.rows) < LOOP_CLOSURE_MIN_GRID_CELLS)
            return result;

        result.inlier_matches.reserve(inliers.rows);
        for (int i = 0; i < inliers.rows; ++i)
            result.inlier_matches.push_back(used_matches[inliers.at<int>(i)]);
        result.valid = true;
        return result;
    }

    VerificationResult landmarkGeometricVerification(const Keyframe &candidate,
                                                      const Keyframe &current)
    {
        VerificationResult result;
        result.uses_landmarks = true;
        if (current.landmark_descriptors.empty() || current.landmarks_world.empty() ||
            candidate.descriptors.empty())
            return result;

        const std::vector<cv::DMatch> matches = LOOP_CLOSURE_MODE == 3
            ? modelMatches(current.landmark_keypoints,
                           current.landmark_descriptors, current.image.size(),
                           candidate.keypoints, candidate.descriptors,
                           candidate.image.size())
            : mutualRatioMatches(current.landmark_descriptors,
                                 candidate.descriptors);

        std::vector<cv::Point3f> object_points;
        std::vector<cv::Point2f> image_points;
        std::vector<cv::Point2f> pixel_points;
        std::vector<cv::DMatch> used_matches;
        std::vector<cv::Point2f> current_pixels;
        std::vector<cv::Point2f> candidate_pixels;
        for (const cv::DMatch &match : matches)
        {
            if (match.queryIdx < 0 || match.queryIdx >= static_cast<int>(current.landmarks_world.size()) ||
                match.trainIdx < 0 || match.trainIdx >= static_cast<int>(candidate.keypoints.size()))
                continue;
            const Eigen::Vector3d &point = current.landmarks_world[match.queryIdx];
            const Eigen::Vector3d candidate_ray = ray(camera_left, candidate.keypoints[match.trainIdx].pt);
            if (!point.allFinite() || std::abs(candidate_ray.z()) < 1e-6)
                continue;
            object_points.emplace_back(point.x(), point.y(), point.z());
            image_points.emplace_back(candidate_ray.x() / candidate_ray.z(),
                                      candidate_ray.y() / candidate_ray.z());
            current_pixels.push_back(current.landmark_keypoints[match.queryIdx].pt);
            candidate_pixels.push_back(candidate.keypoints[match.trainIdx].pt);
            pixel_points.push_back(candidate.keypoints[match.trainIdx].pt);
            used_matches.emplace_back(match.trainIdx, match.queryIdx, match.distance);
        }
        if (static_cast<int>(object_points.size()) < LOOP_CLOSURE_MIN_MATCHES)
            return result;

        std::vector<unsigned char> f_mask;
        if (!fundamentalInlierMask(current_pixels, candidate_pixels, f_mask))
            return result;
        for (int i = static_cast<int>(f_mask.size()) - 1; i >= 0; --i)
        {
            if (f_mask[i])
                continue;
            object_points.erase(object_points.begin() + i);
            image_points.erase(image_points.begin() + i);
            pixel_points.erase(pixel_points.begin() + i);
            current_pixels.erase(current_pixels.begin() + i);
            candidate_pixels.erase(candidate_pixels.begin() + i);
            used_matches.erase(used_matches.begin() + i);
        }
        if (static_cast<int>(object_points.size()) < LOOP_CLOSURE_MIN_MATCHES)
            return result;

        Eigen::Isometry3d t_body_camera = Eigen::Isometry3d::Identity();
        t_body_camera.linear() = RIC[0];
        t_body_camera.translation() = TIC[0];
        const Eigen::Isometry3d t_camera_world = (candidate.raw_pose * t_body_camera).inverse();
        const Eigen::Matrix3d initial_r_camera_world = t_camera_world.linear();
        const Eigen::Vector3d initial_t_camera_world = t_camera_world.translation();
        cv::Mat rotation_cv(3, 3, CV_64F);
        cv::Mat translation_cv(3, 1, CV_64F);
        cv::eigen2cv(initial_r_camera_world, rotation_cv);
        cv::eigen2cv(initial_t_camera_world, translation_cv);
        cv::Mat rvec;
        cv::Rodrigues(rotation_cv, rvec);
        cv::Mat inliers;
        const cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        const double normalized_threshold = LOOP_CLOSURE_PNP_REPROJECTION_ERROR / FOCAL_LENGTH;
        const bool solved = cv::solvePnPRansac(object_points,
                                               image_points,
                                               camera_matrix,
                                               cv::Mat(),
                                               rvec,
                                               translation_cv,
                                               true,
                                               LOOP_CLOSURE_PNP_ITERATIONS,
                                               normalized_threshold,
                                               0.995,
                                               inliers,
                                               cv::SOLVEPNP_ITERATIVE);
        if (!solved || inliers.rows < LOOP_CLOSURE_MIN_INLIERS)
            return result;

        cv::Rodrigues(rvec, rotation_cv);
        Eigen::Matrix3d r_camera_world;
        Eigen::Vector3d t_camera_world_vector;
        cv::cv2eigen(rotation_cv, r_camera_world);
        cv::cv2eigen(translation_cv, t_camera_world_vector);
        Eigen::Isometry3d estimated_t_camera_world = Eigen::Isometry3d::Identity();
        estimated_t_camera_world.linear() = r_camera_world;
        estimated_t_camera_world.translation() = t_camera_world_vector;
        const Eigen::Isometry3d estimated_candidate_pose =
            estimated_t_camera_world.inverse() * t_body_camera.inverse();
        result.relative = estimated_candidate_pose.inverse() * current.raw_pose;
        result.matches = static_cast<int>(object_points.size());
        result.inliers = inliers.rows;
        result.inlier_ratio = static_cast<double>(result.inliers) / result.matches;
        result.inlier_spread = inlierSpread(pixel_points, inliers,
                                            candidate.image.cols, candidate.image.rows);
        estimatePnPUncertainty(object_points, image_points, rvec, translation_cv,
                               inliers, result);
        if (result.relative.translation().norm() > LOOP_CLOSURE_MAX_TRANSLATION ||
            rotationAngle(result.relative.linear()) > LOOP_CLOSURE_MAX_ROTATION_DEG * M_PI / 180.0 ||
            result.inlier_ratio < LOOP_CLOSURE_MIN_INLIER_RATIO ||
            result.inlier_spread < LOOP_CLOSURE_MIN_INLIER_SPREAD ||
            occupiedGridCells(current_pixels, inliers, current.image.cols,
                              current.image.rows) < LOOP_CLOSURE_MIN_GRID_CELLS ||
            occupiedGridCells(candidate_pixels, inliers, candidate.image.cols,
                              candidate.image.rows) < LOOP_CLOSURE_MIN_GRID_CELLS)
            return result;

        result.inlier_matches.reserve(inliers.rows);
        for (int i = 0; i < inliers.rows; ++i)
            result.inlier_matches.push_back(used_matches[inliers.at<int>(i)]);
        result.valid = true;
        return result;
    }

    bool optimize4DoFGraph()
    {
        std::vector<double> yaw_corrections(keyframes.size(), 0.0);
        std::vector<std::array<double, 3>> positions(keyframes.size());
        for (size_t i = 0; i < keyframes.size(); ++i)
        {
            yaw_corrections[i] = normalizedAngle(
                yawAngle(keyframes[i].optimized_pose.linear()) -
                yawAngle(keyframes[i].raw_pose.linear()));
            positions[i] = {keyframes[i].optimized_pose.translation().x(),
                            keyframes[i].optimized_pose.translation().y(),
                            keyframes[i].optimized_pose.translation().z()};
        }

        ceres::Problem problem;
        for (size_t i = 0; i < keyframes.size(); ++i)
        {
            problem.AddParameterBlock(&yaw_corrections[i], 1);
            problem.AddParameterBlock(positions[i].data(), 3);
        }
        problem.SetParameterBlockConstant(&yaw_corrections.front());
        problem.SetParameterBlockConstant(positions.front().data());
        for (const Edge &edge : edges)
        {
            auto *cost = new ceres::AutoDiffCostFunction<FourDoFPoseGraphError, 4, 1, 3, 1, 3>(
                new FourDoFPoseGraphError(edge.transform.translation(),
                                          normalizedAngle(
                                              yawAngle(keyframes[edge.from].raw_pose.linear() *
                                                       edge.transform.linear()) -
                                              yawAngle(keyframes[edge.from].raw_pose.linear())),
                                          keyframes[edge.from].raw_pose.linear(),
                                          keyframes[edge.to].raw_pose.linear(),
                                          edge.translation_weight,
                                          edge.rotation_weight));
            ceres::LossFunction *loss = edge.loop ? new ceres::HuberLoss(1.0) : nullptr;
            problem.AddResidualBlock(cost, loss,
                                     &yaw_corrections[edge.from], positions[edge.from].data(),
                                     &yaw_corrections[edge.to], positions[edge.to].data());
        }
        ceres::Solver::Options options;
        options.max_num_iterations = LOOP_CLOSURE_POSE_GRAPH_ITERATIONS;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = 1;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (!summary.IsSolutionUsable())
        {
            ROS_WARN("independent 4DoF loop graph optimization failed: %s",
                     summary.BriefReport().c_str());
            return false;
        }
        for (size_t i = 0; i < keyframes.size(); ++i)
        {
            const double c = std::cos(yaw_corrections[i]);
            const double s = std::sin(yaw_corrections[i]);
            Eigen::Matrix3d yaw_rotation;
            yaw_rotation << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
            keyframes[i].optimized_pose = makePose(
                Eigen::Vector3d(positions[i][0], positions[i][1], positions[i][2]),
                yaw_rotation * keyframes[i].raw_pose.linear());
        }
        return true;
    }

    bool optimizeGraph()
    {
        if (keyframes.size() < 2)
            return true;
        if (USE_IMU)
            return optimize4DoFGraph();
        std::vector<std::array<double, 7>> poses(keyframes.size());
        for (size_t i = 0; i < keyframes.size(); ++i)
        {
            const Eigen::Quaterniond q(keyframes[i].optimized_pose.linear());
            poses[i] = {keyframes[i].optimized_pose.translation().x(),
                        keyframes[i].optimized_pose.translation().y(),
                        keyframes[i].optimized_pose.translation().z(),
                        q.x(), q.y(), q.z(), q.w()};
        }

        ceres::Problem problem;
        for (size_t i = 0; i < poses.size(); ++i)
            problem.AddParameterBlock(poses[i].data(), 7, new PoseLocalParameterization());
        problem.SetParameterBlockConstant(poses.front().data());
        for (const Edge &edge : edges)
        {
            const Eigen::Quaterniond q(edge.transform.linear());
            auto *cost = new ceres::AutoDiffCostFunction<PoseGraphError, 6, 7, 7>(
                new PoseGraphError(edge.transform.translation(), q,
                                   edge.translation_weight, edge.rotation_weight));
            ceres::LossFunction *loss = edge.loop ? new ceres::HuberLoss(1.0) : nullptr;
            problem.AddResidualBlock(cost, loss, poses[edge.from].data(), poses[edge.to].data());
        }
        ceres::Solver::Options options;
        options.max_num_iterations = LOOP_CLOSURE_POSE_GRAPH_ITERATIONS;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = 1;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (!summary.IsSolutionUsable())
        {
            ROS_WARN("independent loop pose graph optimization failed: %s", summary.BriefReport().c_str());
            return false;
        }

        for (size_t i = 0; i < keyframes.size(); ++i)
        {
            Eigen::Quaterniond q(poses[i][6], poses[i][3], poses[i][4], poses[i][5]);
            keyframes[i].optimized_pose = makePose(
                Eigen::Vector3d(poses[i][0], poses[i][1], poses[i][2]), q.normalized().toRotationMatrix());
        }
        return true;
    }

    void updateCorrection()
    {
        if (keyframes.empty())
            return;
        const Eigen::Isometry3d correction =
            keyframes.back().optimized_pose * keyframes.back().raw_pose.inverse();
        std::lock_guard<std::mutex> lock(mutex);
        r_map_odom = correction.linear();
        t_map_odom = correction.translation();
    }

    bool loopIsPlausibleBeforeOptimization(const Edge &loop) const
    {
        const Eigen::Isometry3d predicted =
            keyframes[loop.from].optimized_pose.inverse() * keyframes[loop.to].optimized_pose;
        const Eigen::Isometry3d error = loop.transform.inverse() * predicted;
        const double rotation_error = USE_IMU
            ? std::abs(yawAngle(error.linear())) : rotationAngle(error.linear());
        return error.translation().norm() <= LOOP_CLOSURE_GRAPH_MAX_TRANSLATION_ERROR &&
               rotation_error <= LOOP_CLOSURE_GRAPH_MAX_ROTATION_ERROR_DEG * M_PI / 180.0;
    }

    bool graphIsConsistent(double &max_translation_error,
                           double &max_rotation_error_deg,
                           double &max_normalized_error) const
    {
        max_translation_error = 0.0;
        max_rotation_error_deg = 0.0;
        max_normalized_error = 0.0;
        for (const Edge &edge : edges)
        {
            const Eigen::Isometry3d predicted =
                keyframes[edge.from].optimized_pose.inverse() * keyframes[edge.to].optimized_pose;
            const Eigen::Isometry3d error = edge.transform.inverse() * predicted;
            max_translation_error = std::max(max_translation_error, error.translation().norm());
            const double rotation_error = USE_IMU
                ? std::abs(yawAngle(error.linear())) : rotationAngle(error.linear());
            max_rotation_error_deg = std::max(max_rotation_error_deg,
                                              rotation_error * 180.0 / M_PI);
            max_normalized_error = std::max(max_normalized_error,
                std::max(error.translation().norm() /
                             std::max(1e-6, edge.translation_std),
                         rotation_error / std::max(1e-6, edge.rotation_std)));
        }
        return max_translation_error <= LOOP_CLOSURE_GRAPH_MAX_TRANSLATION_ERROR &&
               max_rotation_error_deg <= LOOP_CLOSURE_GRAPH_MAX_ROTATION_ERROR_DEG &&
               max_normalized_error <= LOOP_CLOSURE_GRAPH_MAX_NORMALIZED_ERROR;
    }

    void publishGraph(const Keyframe *candidate,
                      const Keyframe *current,
                      const std::vector<cv::DMatch> &matches,
                      bool landmark_matches = false)
    {
        if (!path_publisher || keyframes.empty())
            return;
        nav_msgs::msg::Path graph_path;
        graph_path.header.frame_id = WORLD_FRAME_ID;
        const double stamp = keyframes.back().stamp;
        graph_path.header.stamp.sec = static_cast<int32_t>(stamp);
        graph_path.header.stamp.nanosec = static_cast<uint32_t>((stamp - std::floor(stamp)) * 1e9);
        for (const Keyframe &keyframe : keyframes)
        {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = graph_path.header;
            pose.pose.position.x = keyframe.optimized_pose.translation().x();
            pose.pose.position.y = keyframe.optimized_pose.translation().y();
            pose.pose.position.z = keyframe.optimized_pose.translation().z();
            const Eigen::Quaterniond q(keyframe.optimized_pose.linear());
            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();
            pose.pose.orientation.w = q.w();
            graph_path.poses.push_back(pose);
        }
        path_publisher->publish(graph_path);

        visualization_msgs::msg::Marker marker;
        marker.header = graph_path.header;
        marker.ns = "independent_loop_edges";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.04;
        marker.color.r = 1.0;
        marker.color.g = 0.1;
        marker.color.b = 0.1;
        marker.color.a = 1.0;
        for (const Edge &edge : edges)
        {
            if (!edge.loop)
                continue;
            geometry_msgs::msg::Point a;
            geometry_msgs::msg::Point b;
            a.x = keyframes[edge.from].optimized_pose.translation().x();
            a.y = keyframes[edge.from].optimized_pose.translation().y();
            a.z = keyframes[edge.from].optimized_pose.translation().z();
            b.x = keyframes[edge.to].optimized_pose.translation().x();
            b.y = keyframes[edge.to].optimized_pose.translation().y();
            b.z = keyframes[edge.to].optimized_pose.translation().z();
            marker.points.push_back(a);
            marker.points.push_back(b);
        }
        edge_publisher->publish(marker);

        if (candidate && current && !matches.empty() && match_publisher)
        {
            cv::Mat output;
            cv::drawMatches(candidate->image, candidate->keypoints,
                            current->image,
                            landmark_matches ? current->landmark_keypoints : current->keypoints,
                            matches, output,
                            cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255));
            match_publisher->publish(*cv_bridge::CvImage(graph_path.header, "bgr8", output).toImageMsg());
        }
    }

    void process(Input input)
    {
        std::lock_guard<std::mutex> database_lock(database_mutex);
        Keyframe current;
        current.id = static_cast<int>(keyframes.size());
        current.stamp = input.stamp;
        current.image = input.left;
        current.raw_pose = input.raw_pose;
        Eigen::Matrix3d correction_r;
        Eigen::Vector3d correction_t;
        {
            std::lock_guard<std::mutex> lock(mutex);
            correction_r = r_map_odom;
            correction_t = t_map_odom;
        }
        current.optimized_pose = makePose(
            correction_r * input.raw_pose.translation() + correction_t,
            correction_r * input.raw_pose.linear());
        extract(current, input.right, input.landmarks_world, input.landmark_pixels,
                input.landmark_ids, input.model_features);
        if (current.descriptors.empty())
        {
            ROS_WARN("independent loop: keyframe %d has no %s descriptors", current.id,
                     LOOP_CLOSURE_MODE == 3 ? "YOLOPoint" : "ORB");
            return;
        }

        double score = 0.0;
        const int candidate_id = detectCandidate(current, score);
        VerificationResult verification;
        if (candidate_id >= 0 && current.id - last_verification_keyframe >= LOOP_CLOSURE_MIN_INTERVAL)
        {
            last_verification_keyframe = current.id;
            const VerificationResult landmark =
                landmarkGeometricVerification(keyframes[candidate_id], current);
            const VerificationResult stereo =
                stereoGeometricVerification(keyframes[candidate_id], current);
            if (landmark.valid && stereo.valid)
            {
                const Eigen::Isometry3d disagreement = landmark.relative.inverse() * stereo.relative;
                if (disagreement.translation().norm() <= LOOP_CLOSURE_DUAL_MAX_TRANSLATION_ERROR &&
                    rotationAngle(disagreement.linear()) <=
                        LOOP_CLOSURE_DUAL_MAX_ROTATION_ERROR_DEG * M_PI / 180.0)
                    verification = landmark;
                else
                    ROS_WARN("independent loop dual verification disagreed: translation %.3f m, rotation %.2f deg",
                             disagreement.translation().norm(),
                             rotationAngle(disagreement.linear()) * 180.0 / M_PI);
            }
            else if (landmark.valid)
                verification = landmark;
            else if (stereo.valid && LOOP_CLOSURE_ALLOW_STEREO_FALLBACK)
                verification = stereo;
        }

        const int neighbor_count = std::min(LOOP_CLOSURE_NEIGHBOR_EDGES,
                                            static_cast<int>(keyframes.size()));
        for (int offset = neighbor_count; offset >= 1; --offset)
        {
            const Keyframe &previous = keyframes[keyframes.size() - offset];
            Edge neighbor;
            neighbor.from = previous.id;
            neighbor.to = current.id;
            neighbor.transform = previous.raw_pose.inverse() * current.raw_pose;
            neighbor.translation_weight = LOOP_CLOSURE_ODOM_TRANSLATION_WEIGHT;
            neighbor.rotation_weight = LOOP_CLOSURE_ODOM_ROTATION_WEIGHT;
            neighbor.translation_std = 1.0 / LOOP_CLOSURE_ODOM_TRANSLATION_WEIGHT;
            neighbor.rotation_std = 1.0 / LOOP_CLOSURE_ODOM_ROTATION_WEIGHT;
            edges.push_back(neighbor);
        }
        addDocument(current);
        keyframes.push_back(std::move(current));

        if (verification.valid)
        {
            Edge loop;
            loop.from = candidate_id;
            loop.to = keyframes.back().id;
            loop.transform = verification.relative;
            const double quality = std::max(0.25,
                0.5 * std::min(1.0, verification.inlier_ratio / 0.70) +
                0.5 * std::min(1.0, verification.inlier_spread / 0.40));
            loop.translation_std = verification.translation_std;
            loop.rotation_std = verification.rotation_std;
            const double translation_confidence = std::sqrt(
                LOOP_CLOSURE_MIN_TRANSLATION_STD / loop.translation_std);
            const double rotation_confidence = std::sqrt(
                (LOOP_CLOSURE_MIN_ROTATION_STD_DEG * M_PI / 180.0) /
                loop.rotation_std);
            loop.translation_weight = LOOP_CLOSURE_LOOP_TRANSLATION_WEIGHT *
                                      std::sqrt(quality) * translation_confidence;
            loop.rotation_weight = LOOP_CLOSURE_LOOP_ROTATION_WEIGHT *
                                   std::sqrt(quality) * rotation_confidence;
            loop.loop = true;
            if (!loopIsPlausibleBeforeOptimization(loop))
            {
                ROS_WARN("independent loop rejected before optimization: %d <-> %d",
                         candidate_id, keyframes.back().id);
                publishGraph(nullptr, nullptr, {});
            }
            else
            {
                std::vector<Eigen::Isometry3d> pose_snapshot;
                pose_snapshot.reserve(keyframes.size());
                for (const Keyframe &keyframe : keyframes)
                    pose_snapshot.push_back(keyframe.optimized_pose);
                edges.push_back(loop);
                double max_translation_error = 0.0;
                double max_rotation_error_deg = 0.0;
                double max_normalized_error = 0.0;
                const bool optimized = optimizeGraph();
                const bool consistent = optimized &&
                    graphIsConsistent(max_translation_error, max_rotation_error_deg,
                                      max_normalized_error);
                if (!consistent)
                {
                    edges.pop_back();
                    for (size_t i = 0; i < keyframes.size(); ++i)
                        keyframes[i].optimized_pose = pose_snapshot[i];
                    ROS_WARN("independent loop rolled back: %d <-> %d, graph max error %.3f m / %.2f deg / %.2f sigma",
                             candidate_id, keyframes.back().id,
                             max_translation_error, max_rotation_error_deg,
                             max_normalized_error);
                    publishGraph(nullptr, nullptr, {});
                }
                else
                {
                    updateCorrection();
                    ++accepted_loops;
                    ROS_INFO("independent loop accepted: %d <-> %d, score %.3f, %s PnP %d/%d inliers, spread %.3f, reproj %.2f px, depth %.2f m, sigma %.3f m/%.2f deg, weights %.2f/%.2f, total %d",
                             candidate_id, keyframes.back().id, score,
                             verification.uses_landmarks ? "landmark" : "stereo",
                             verification.inliers, verification.matches,
                             verification.inlier_spread,
                             verification.reprojection_rmse_px,
                             verification.median_depth,
                             verification.translation_std,
                             verification.rotation_std * 180.0 / M_PI,
                             loop.translation_weight, loop.rotation_weight,
                             accepted_loops);
                    publishGraph(&keyframes[candidate_id], &keyframes.back(),
                                 verification.inlier_matches,
                                 verification.uses_landmarks);
                }
            }
        }
        else
        {
            publishGraph(nullptr, nullptr, {});
        }
    }
};

IndependentLoopClosure::IndependentLoopClosure() : impl_(new Impl)
{
}

IndependentLoopClosure::~IndependentLoopClosure() = default;

void IndependentLoopClosure::init(const rclcpp::Node::SharedPtr &node)
{
    impl_->stop();
    if (!LOOP_CLOSURE_ENABLE)
    {
        ROS_INFO("independent loop closure: disabled by YAML");
        return;
    }
    if (!STEREO || CAM_NAMES.size() < 2 || RIC.size() < 2 || TIC.size() < 2)
    {
        ROS_WARN("independent loop closure requires stereo calibration; disabling");
        LOOP_CLOSURE_ENABLE = 0;
        return;
    }
    if (LOOP_CLOSURE_MODE == 3 && FEATURE_TRACKER_TYPE != 3)
    {
        ROS_ERROR("loop_closure_mode 3 requires feature_tracker_type 3; disabling loop closure");
        LOOP_CLOSURE_ENABLE = 0;
        return;
    }
    impl_->camera_left = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(CAM_NAMES[0]);
    impl_->camera_right = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(CAM_NAMES[1]);
    if (!impl_->camera_left || !impl_->camera_right)
    {
        ROS_ERROR("independent loop closure cannot load camera calibration; disabling");
        LOOP_CLOSURE_ENABLE = 0;
        return;
    }
    if (LOOP_CLOSURE_MODE == 1)
        impl_->orb = cv::ORB::create(LOOP_CLOSURE_MAX_FEATURES);
    else
    {
        try
        {
            impl_->initializeModelMatcher();
        }
        catch (const std::exception &error)
        {
            ROS_ERROR("independent loop cannot initialize LightGlue: %s; disabling",
                      error.what());
            LOOP_CLOSURE_ENABLE = 0;
            return;
        }
    }
    impl_->path_publisher = node->create_publisher<nav_msgs::msg::Path>("~/loop_path", 10);
    impl_->edge_publisher = node->create_publisher<visualization_msgs::msg::Marker>("~/loop_edges", 10);
    impl_->match_publisher = node->create_publisher<sensor_msgs::msg::Image>("~/loop_match_image", 2);
    impl_->running.store(true);
    impl_->worker = std::thread([this] { impl_->run(); });
    impl_->submitted_keyframes.store(0);
    ROS_INFO("independent loop closure: enabled, mode=%d (%s), features=%d, stride=%d, STM=%d, threshold=%.3f",
             LOOP_CLOSURE_MODE,
             LOOP_CLOSURE_MODE == 3 ? "YOLOPoint+LightGlue" : "ORB",
             LOOP_CLOSURE_MODE == 3 ? YOLOPOINT_LIGHTGLUE_MAX_KEYPOINTS
                                    : LOOP_CLOSURE_MAX_FEATURES,
             LOOP_CLOSURE_KEYFRAME_STRIDE, LOOP_CLOSURE_STM_SIZE,
             LOOP_CLOSURE_MODE == 3 ? LOOP_CLOSURE_MODEL_APPEARANCE_THRESHOLD
                                    : LOOP_CLOSURE_APPEARANCE_THRESHOLD);
}

void IndependentLoopClosure::reset()
{
    std::lock_guard<std::mutex> database_lock(impl_->database_mutex);
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->queue.clear();
    impl_->keyframes.clear();
    impl_->edges.clear();
    {
        std::lock_guard<std::mutex> trajectory_lock(impl_->trajectory_mutex);
        impl_->odometry_samples.clear();
    }
    impl_->word_document_frequency.clear();
    impl_->r_map_odom.setIdentity();
    impl_->t_map_odom.setZero();
    impl_->last_candidate = -1;
    impl_->consistent_hits = 0;
    impl_->accepted_loops = 0;
    impl_->last_verification_keyframe = -1000000;
    impl_->submitted_keyframes.store(0);
}

void IndependentLoopClosure::shutdown()
{
    // The publishers belong to the ROS node's DDS participant.  Release them
    // while the node is still alive instead of leaving them in this process-
    // lifetime singleton, whose destructor runs after main() has returned.
    impl_->stop();
    impl_->model_matcher_session.reset();
    impl_->orb.release();
    impl_->match_publisher.reset();
    impl_->edge_publisher.reset();
    impl_->path_publisher.reset();
}

void IndependentLoopClosure::submitKeyframe(double stamp,
                                            const cv::Mat &left,
                                            const cv::Mat &right,
                                            const Eigen::Vector3d &p_odom_body,
                                            const Eigen::Matrix3d &r_odom_body,
                                            const std::vector<Eigen::Vector3d> &landmarks_world,
                                            const std::vector<cv::Point2f> &landmark_pixels,
                                            const std::vector<int> &landmark_ids,
                                            const LoopModelFeatures &model_features)
{
    if (!LOOP_CLOSURE_ENABLE || !impl_->running.load() || left.empty() || right.empty())
        return;
    const uint64_t sequence = impl_->submitted_keyframes.fetch_add(1);
    if (sequence % static_cast<uint64_t>(LOOP_CLOSURE_KEYFRAME_STRIDE) != 0)
        return;
    Impl::Input input;
    input.stamp = stamp;
    input.left = left.clone();
    input.right = right.clone();
    input.raw_pose = makePose(p_odom_body, r_odom_body);
    input.landmarks_world = landmarks_world;
    input.landmark_pixels = landmark_pixels;
    input.landmark_ids = landmark_ids;
    input.model_features = model_features;
    input.model_features.descriptors = model_features.descriptors.clone();
    input.model_features.global_descriptor = model_features.global_descriptor.clone();
    if (!MULTIPLE_THREAD)
    {
        impl_->process(std::move(input));
        return;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->queue.size() >= static_cast<size_t>(LOOP_CLOSURE_MAX_PENDING_KEYFRAMES))
        {
            impl_->queue.pop_front();
            ROS_WARN("independent loop queue full, dropping oldest pending keyframe");
        }
        impl_->queue.push_back(std::move(input));
    }
    impl_->condition.notify_one();
}

void IndependentLoopClosure::correctPose(const Eigen::Vector3d &p_odom_body,
                                         const Eigen::Quaterniond &q_odom_body,
                                         Eigen::Vector3d &p_map_body,
                                         Eigen::Quaterniond &q_map_body) const
{
    if (!LOOP_CLOSURE_ENABLE)
    {
        p_map_body = p_odom_body;
        q_map_body = q_odom_body;
        return;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    p_map_body = impl_->r_map_odom * p_odom_body + impl_->t_map_odom;
    q_map_body = Eigen::Quaterniond(impl_->r_map_odom) * q_odom_body;
    q_map_body.normalize();
}

Eigen::Vector3d IndependentLoopClosure::correctVector(const Eigen::Vector3d &v_odom) const
{
    if (!LOOP_CLOSURE_ENABLE)
        return v_odom;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->r_map_odom * v_odom;
}

void IndependentLoopClosure::recordOdometry(
    double stamp,
    const Eigen::Vector3d &p_odom_body,
    const Eigen::Quaterniond &q_odom_body,
    const Eigen::Vector3d &v_odom_body)
{
    if (!LOOP_CLOSURE_ENABLE || !impl_->running.load())
        return;
    Impl::OdometrySample sample;
    sample.stamp = stamp;
    sample.raw_pose = makePose(p_odom_body, q_odom_body.normalized().toRotationMatrix());
    sample.raw_velocity = v_odom_body;
    std::lock_guard<std::mutex> lock(impl_->trajectory_mutex);
    if (!impl_->odometry_samples.empty() &&
        stamp <= impl_->odometry_samples.back().stamp)
        return;
    impl_->odometry_samples.push_back(std::move(sample));
}

void IndependentLoopClosure::waitUntilIdle()
{
    if (!LOOP_CLOSURE_ENABLE || !impl_->running.load() || !MULTIPLE_THREAD)
        return;
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->idle_condition.wait(lock, [this] {
        return impl_->queue.empty() && !impl_->processing;
    });
}

bool IndependentLoopClosure::saveTrajectory(const std::string &path)
{
    waitUntilIdle();
    std::lock_guard<std::mutex> database_lock(impl_->database_mutex);
    if (impl_->keyframes.empty())
    {
        ROS_WARN("independent loop cannot save trajectory: no keyframes");
        return false;
    }
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output.is_open())
    {
        ROS_ERROR("independent loop cannot save trajectory to %s", path.c_str());
        return false;
    }
    output.setf(std::ios::fixed, std::ios::floatfield);
    output.precision(9);
    for (const Impl::Keyframe &keyframe : impl_->keyframes)
    {
        const Eigen::Quaterniond q(keyframe.optimized_pose.linear());
        output << keyframe.stamp << " "
               << keyframe.optimized_pose.translation().x() << " "
               << keyframe.optimized_pose.translation().y() << " "
               << keyframe.optimized_pose.translation().z() << " "
               << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    ROS_INFO("independent loop saved %zu optimized keyframes (%d accepted loops) to %s",
             impl_->keyframes.size(), impl_->accepted_loops, path.c_str());
    return !impl_->keyframes.empty();
}

bool IndependentLoopClosure::saveOptimizedTrajectory(const std::string &path)
{
    waitUntilIdle();
    std::lock_guard<std::mutex> database_lock(impl_->database_mutex);
    std::lock_guard<std::mutex> trajectory_lock(impl_->trajectory_mutex);
    if (impl_->keyframes.empty() || impl_->odometry_samples.empty())
    {
        ROS_WARN("independent loop cannot save dense optimized trajectory: keyframes=%zu samples=%zu",
                 impl_->keyframes.size(), impl_->odometry_samples.size());
        return false;
    }

    std::vector<Eigen::Isometry3d> corrections;
    corrections.reserve(impl_->keyframes.size());
    for (const Impl::Keyframe &keyframe : impl_->keyframes)
    {
        // The keyframe raw pose is captured after it has crossed the VINS
        // sliding window, while the online sample at the same timestamp was
        // published earlier.  Build the correction against that actual
        // published raw sample so the dense result passes through every
        // optimized keyframe instead of mixing two estimator revisions.
        auto upper = std::lower_bound(
            impl_->odometry_samples.begin(), impl_->odometry_samples.end(),
            keyframe.stamp,
            [](const Impl::OdometrySample &sample, double stamp) {
                return sample.stamp < stamp;
            });
        const Impl::OdometrySample *nearest = nullptr;
        if (upper == impl_->odometry_samples.begin())
            nearest = &*upper;
        else if (upper == impl_->odometry_samples.end())
            nearest = &impl_->odometry_samples.back();
        else
        {
            const Impl::OdometrySample &before = *(upper - 1);
            nearest = std::abs(before.stamp - keyframe.stamp) <=
                              std::abs(upper->stamp - keyframe.stamp)
                          ? &before : &*upper;
        }
        const Eigen::Isometry3d &raw_reference =
            nearest && std::abs(nearest->stamp - keyframe.stamp) < 0.06
                ? nearest->raw_pose : keyframe.raw_pose;
        corrections.push_back(keyframe.optimized_pose * raw_reference.inverse());
    }

    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output.is_open())
    {
        ROS_ERROR("independent loop cannot save dense optimized trajectory to %s", path.c_str());
        return false;
    }
    output.setf(std::ios::fixed, std::ios::floatfield);
    size_t upper = 0;
    for (const Impl::OdometrySample &sample : impl_->odometry_samples)
    {
        while (upper < impl_->keyframes.size() &&
               impl_->keyframes[upper].stamp < sample.stamp)
            ++upper;

        Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();
        if (upper == 0)
            correction = corrections.front();
        else if (upper >= impl_->keyframes.size())
            correction = corrections.back();
        else
        {
            const size_t lower = upper - 1;
            const double duration = impl_->keyframes[upper].stamp -
                                    impl_->keyframes[lower].stamp;
            const double alpha = duration > 1e-9
                ? std::min(1.0, std::max(0.0,
                    (sample.stamp - impl_->keyframes[lower].stamp) / duration))
                : 1.0;
            const Eigen::Quaterniond lower_q(corrections[lower].linear());
            Eigen::Quaterniond upper_q(corrections[upper].linear());
            if (lower_q.dot(upper_q) < 0.0)
                upper_q.coeffs() *= -1.0;
            correction.linear() = lower_q.slerp(alpha, upper_q).normalized().toRotationMatrix();
            correction.translation() =
                (1.0 - alpha) * corrections[lower].translation() +
                alpha * corrections[upper].translation();
        }

        const Eigen::Isometry3d optimized = correction * sample.raw_pose;
        const Eigen::Quaterniond q(optimized.linear());
        const Eigen::Vector3d velocity = correction.linear() * sample.raw_velocity;
        output.precision(9);
        output << sample.stamp << ",";
        output.precision(5);
        output << optimized.translation().x() << ","
               << optimized.translation().y() << ","
               << optimized.translation().z() << ","
               << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ","
               << velocity.x() << "," << velocity.y() << "," << velocity.z() << ",\n";
    }
    ROS_INFO("independent loop saved %zu uniformly optimized poses to %s",
             impl_->odometry_samples.size(), path.c_str());
    return true;
}

bool IndependentLoopClosure::enabled() const
{
    return LOOP_CLOSURE_ENABLE && impl_->running.load();
}

IndependentLoopClosure &independentLoopClosure()
{
    static IndependentLoopClosure instance;
    return instance;
}

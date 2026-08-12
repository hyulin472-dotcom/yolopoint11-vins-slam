#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>
#include <vector>

#include "loop_model_features.h"

class IndependentLoopClosure
{
public:
    IndependentLoopClosure();
    ~IndependentLoopClosure();

    void init(const rclcpp::Node::SharedPtr &node);
    void reset();
    void shutdown();

    void submitKeyframe(double stamp,
                        const cv::Mat &left,
                        const cv::Mat &right,
                        const Eigen::Vector3d &p_odom_body,
                        const Eigen::Matrix3d &r_odom_body,
                        const std::vector<Eigen::Vector3d> &landmarks_world,
                        const std::vector<cv::Point2f> &landmark_pixels,
                        const std::vector<int> &landmark_ids,
                        const LoopModelFeatures &model_features);

    void correctPose(const Eigen::Vector3d &p_odom_body,
                     const Eigen::Quaterniond &q_odom_body,
                     Eigen::Vector3d &p_map_body,
                     Eigen::Quaterniond &q_map_body) const;

    Eigen::Vector3d correctVector(const Eigen::Vector3d &v_odom) const;
    void recordOdometry(double stamp,
                        const Eigen::Vector3d &p_odom_body,
                        const Eigen::Quaterniond &q_odom_body,
                        const Eigen::Vector3d &v_odom_body);
    void waitUntilIdle();
    bool saveTrajectory(const std::string &path);
    bool saveOptimizedTrajectory(const std::string &path);
    bool enabled() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

IndependentLoopClosure &independentLoopClosure();

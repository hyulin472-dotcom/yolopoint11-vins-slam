#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <vector>

// Immutable snapshot of the YOLOPointv11 features needed by loop closure.
// The frontend creates it from the same inference used by feature_tracker_type
// 3, so enabling model loop closure does not run the extractor a second time.
struct LoopModelFeatures
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;       // N x 128, CV_32F, row-normalized
    cv::Mat global_descriptor; // 1 x 128, CV_32F, normalized
    std::vector<int> feature_ids;
    std::vector<cv::Point2f> right_points;
    std::vector<unsigned char> has_right;

    bool valid() const
    {
        return !global_descriptor.empty() &&
               descriptors.type() == CV_32FC1 &&
               descriptors.rows == static_cast<int>(keypoints.size()) &&
               feature_ids.size() == keypoints.size() &&
               right_points.size() == keypoints.size() &&
               has_right.size() == keypoints.size();
    }
};

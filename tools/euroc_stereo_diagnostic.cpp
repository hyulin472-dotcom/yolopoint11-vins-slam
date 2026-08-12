#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <rclcpp/rclcpp.hpp>

#include "../vins/src/estimator/parameters.h"
#include "../vins/src/featureTracker/yolopoint_lightglue_feature_tracker.h"
#include "camodocal/camera_models/CameraFactory.h"

namespace
{
struct RawStats
{
    long input = 0;
    long forward = 0;
    long reverse = 0;
    long fb05 = 0;
    long fb10 = 0;
    long fb15 = 0;
    long epipolar15 = 0;
    long epipolar30 = 0;
    long horizontal_disparity = 0;
    long positive_depth = 0;
    long negative_disparity_positive_depth = 0;
};

struct RectifiedStats
{
    long input = 0;
    long mapped = 0;
    long forward = 0;
    long reverse = 0;
    long fb05 = 0;
    long fb10 = 0;
    long row15 = 0;
    long positive_disparity = 0;
};

struct TemporalConsistencyStats
{
    long transitions = 0;
    long previous_points = 0;
    long lightglue_inherited = 0;
    long klt_fb_valid = 0;
    long inherited_with_klt_fb = 0;
    long agreement05 = 0;
    long agreement10 = 0;
    long agreement20 = 0;
    long agreement30 = 0;
    long agreement50 = 0;
    std::vector<double> coordinate_errors;
};

struct StereoCorrespondenceStats
{
    long input = 0;
    long raw_fb_valid = 0;
    long rectified_fb_valid = 0;
    long both_valid = 0;
    long agreement05 = 0;
    long agreement10 = 0;
    long agreement20 = 0;
    long agreement30 = 0;
    long agreement50 = 0;
    std::vector<double> right_pixel_errors;
};

struct StereoDepthStats
{
    long stereo_observations = 0;
    long inherited_observations = 0;
    long temporal_klt_comparable = 0;
    std::vector<double> depths;
    std::vector<double> rectified_disparities;
    std::vector<double> parallax_degrees;
    std::vector<double> temporal_left_shifts;
    std::vector<double> temporal_disparity_changes;
    std::vector<double> temporal_relative_depth_changes;
};

double percentage(long value, long denominator)
{
    return denominator > 0 ? 100.0 * value / denominator : 0.0;
}

bool inBorder(const cv::Point2f &point, int width, int height)
{
    const int x = cvRound(point.x);
    const int y = cvRound(point.y);
    return x >= 1 && y >= 1 && x < width - 1 && y < height - 1;
}

class StereoGeometry
{
  public:
    StereoGeometry(
        const camodocal::CameraPtr &camera0,
        const camodocal::CameraPtr &camera1,
        int width,
        int height)
        : cameras_{camera0, camera1}, width_(width), height_(height)
    {
        rotation_10_ = RIC[1].transpose() * RIC[0];
        translation_10_ = RIC[1].transpose() * (TIC[0] - TIC[1]);
        Eigen::Matrix3d translation_skew;
        translation_skew <<
            0.0, -translation_10_.z(), translation_10_.y(),
            translation_10_.z(), 0.0, -translation_10_.x(),
            -translation_10_.y(), translation_10_.x(), 0.0;
        essential_ = translation_skew * rotation_10_;
        buildRectification();
    }

    void rectify(
        const cv::Mat &left,
        const cv::Mat &right,
        cv::Mat &rectified_left,
        cv::Mat &rectified_right) const
    {
        cv::remap(
            left, rectified_left, map0_x_, map0_y_,
            cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        cv::remap(
            right, rectified_right, map1_x_, map1_y_,
            cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    }

    bool rawLeftToRectified(
        const cv::Point2f &raw,
        cv::Point2f &rectified) const
    {
        Eigen::Vector3d ray;
        cameras_[0]->liftProjective(
            Eigen::Vector2d(raw.x, raw.y), ray);
        const Eigen::Vector3d rectified_ray = rectification0_ * ray;
        if (rectified_ray.z() <= 1e-9)
            return false;
        rectified.x = static_cast<float>(
            focal_ * rectified_ray.x() / rectified_ray.z() + cx_);
        rectified.y = static_cast<float>(
            focal_ * rectified_ray.y() / rectified_ray.z() + cy_);
        return inBorder(rectified, width_, height_);
    }

    bool rectifiedRightToRaw(
        const cv::Point2f &rectified,
        cv::Point2f &raw) const
    {
        const Eigen::Vector3d rectified_ray(
            (rectified.x - cx_) / focal_,
            (rectified.y - cy_) / focal_,
            1.0);
        Eigen::Vector2d raw_point;
        cameras_[1]->spaceToPlane(
            rectification1_.transpose() * rectified_ray, raw_point);
        raw.x = static_cast<float>(raw_point.x());
        raw.y = static_cast<float>(raw_point.y());
        return inBorder(raw, width_, height_);
    }

    bool rawRightToRectified(
        const cv::Point2f &raw,
        cv::Point2f &rectified) const
    {
        Eigen::Vector3d ray;
        cameras_[1]->liftProjective(
            Eigen::Vector2d(raw.x, raw.y), ray);
        const Eigen::Vector3d rectified_ray = rectification1_ * ray;
        if (rectified_ray.z() <= 1e-9)
            return false;
        rectified.x = static_cast<float>(
            focal_ * rectified_ray.x() / rectified_ray.z() + cx_);
        rectified.y = static_cast<float>(
            focal_ * rectified_ray.y() / rectified_ray.z() + cy_);
        return inBorder(rectified, width_, height_);
    }

    bool triangulate(
        const cv::Point2f &left,
        const cv::Point2f &right,
        double &left_depth,
        double &right_depth,
        double &parallax_degrees) const
    {
        Eigen::Vector3d left_ray, right_ray;
        cameras_[0]->liftProjective(
            Eigen::Vector2d(left.x, left.y), left_ray);
        cameras_[1]->liftProjective(
            Eigen::Vector2d(right.x, right.y), right_ray);
        if (std::abs(left_ray.z()) <= 1e-12 ||
            std::abs(right_ray.z()) <= 1e-12)
            return false;
        left_ray /= left_ray.z();
        right_ray /= right_ray.z();
        Eigen::Matrix<double, 3, 2> system;
        system.col(0) = rotation_10_ * left_ray;
        system.col(1) = -right_ray;
        const Eigen::Vector2d depths =
            system.colPivHouseholderQr().solve(-translation_10_);
        left_depth = depths.x();
        right_depth = depths.y();
        const Eigen::Vector3d left_in_right =
            (rotation_10_ * left_ray).normalized();
        const Eigen::Vector3d right_unit = right_ray.normalized();
        const double cosine = std::clamp(
            left_in_right.dot(right_unit), -1.0, 1.0);
        parallax_degrees =
            std::acos(cosine) * 180.0 / 3.14159265358979323846;
        return std::isfinite(left_depth) && std::isfinite(right_depth) &&
               std::isfinite(parallax_degrees);
    }

    double epipolarPixelError(
        const cv::Point2f &left,
        const cv::Point2f &right) const
    {
        Eigen::Vector3d left_ray, right_ray;
        cameras_[0]->liftProjective(
            Eigen::Vector2d(left.x, left.y), left_ray);
        cameras_[1]->liftProjective(
            Eigen::Vector2d(right.x, right.y), right_ray);
        left_ray /= left_ray.z();
        right_ray /= right_ray.z();
        const Eigen::Vector3d line = essential_ * left_ray;
        const double denominator =
            std::sqrt(line.x() * line.x() + line.y() * line.y());
        if (denominator <= 1e-12)
            return std::numeric_limits<double>::infinity();
        return std::abs(right_ray.dot(line)) / denominator * FOCAL_LENGTH;
    }

    bool hasPositiveDepth(
        const cv::Point2f &left,
        const cv::Point2f &right) const
    {
        Eigen::Vector3d left_ray, right_ray;
        cameras_[0]->liftProjective(
            Eigen::Vector2d(left.x, left.y), left_ray);
        cameras_[1]->liftProjective(
            Eigen::Vector2d(right.x, right.y), right_ray);
        left_ray.normalize();
        right_ray.normalize();
        Eigen::Matrix<double, 3, 2> system;
        system.col(0) = rotation_10_ * left_ray;
        system.col(1) = -right_ray;
        const Eigen::Vector2d depths =
            system.colPivHouseholderQr().solve(-translation_10_);
        return depths.x() > 0.05 && depths.y() > 0.05 &&
               depths.x() < 200.0 && depths.y() < 200.0;
    }

  private:
    void buildRectification()
    {
        const Eigen::Vector3d baseline0 =
            RIC[0].transpose() * (TIC[1] - TIC[0]);
        Eigen::Vector3d x_axis = baseline0.normalized();
        const Eigen::Vector3d camera1_z_in_camera0 =
            rotation_10_.transpose() * Eigen::Vector3d::UnitZ();
        Eigen::Vector3d z_axis =
            Eigen::Vector3d::UnitZ() + camera1_z_in_camera0;
        z_axis -= x_axis * x_axis.dot(z_axis);
        z_axis.normalize();
        Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
        z_axis = x_axis.cross(y_axis).normalized();
        rectification0_.row(0) = x_axis.transpose();
        rectification0_.row(1) = y_axis.transpose();
        rectification0_.row(2) = z_axis.transpose();
        rectification1_ = rectification0_ * rotation_10_.transpose();

        map0_x_.create(height_, width_, CV_32FC1);
        map0_y_.create(height_, width_, CV_32FC1);
        map1_x_.create(height_, width_, CV_32FC1);
        map1_y_.create(height_, width_, CV_32FC1);
        for (int y = 0; y < height_; ++y)
        {
            for (int x = 0; x < width_; ++x)
            {
                const Eigen::Vector3d rectified_ray(
                    (x - cx_) / focal_, (y - cy_) / focal_, 1.0);
                Eigen::Vector2d raw0, raw1;
                cameras_[0]->spaceToPlane(
                    rectification0_.transpose() * rectified_ray, raw0);
                cameras_[1]->spaceToPlane(
                    rectification1_.transpose() * rectified_ray, raw1);
                map0_x_.at<float>(y, x) = static_cast<float>(raw0.x());
                map0_y_.at<float>(y, x) = static_cast<float>(raw0.y());
                map1_x_.at<float>(y, x) = static_cast<float>(raw1.x());
                map1_y_.at<float>(y, x) = static_cast<float>(raw1.y());
            }
        }
    }

    std::vector<camodocal::CameraPtr> cameras_;
    int width_;
    int height_;
    const double focal_ = 460.0;
    double cx_ = 376.0;
    double cy_ = 240.0;
    Eigen::Matrix3d rotation_10_;
    Eigen::Vector3d translation_10_;
    Eigen::Matrix3d essential_;
    Eigen::Matrix3d rectification0_;
    Eigen::Matrix3d rectification1_;
    cv::Mat map0_x_, map0_y_, map1_x_, map1_y_;
};

void runKlt(
    const cv::Mat &left,
    const cv::Mat &right,
    const std::vector<cv::Point2f> &left_points,
    std::vector<cv::Point2f> &right_points,
    std::vector<cv::Point2f> &reverse_points,
    std::vector<unsigned char> &forward_status,
    std::vector<unsigned char> &reverse_status)
{
    if (left_points.empty())
        return;
    std::vector<float> errors;
    const cv::Size window(21, 21);
    cv::calcOpticalFlowPyrLK(
        left, right, left_points, right_points,
        forward_status, errors, window,
        3);
    cv::calcOpticalFlowPyrLK(
        right, left, right_points, reverse_points,
        reverse_status, errors, window,
        3);
}

void analyzeRaw(
    const cv::Mat &left,
    const cv::Mat &right,
    const std::vector<cv::Point2f> &left_points,
    const StereoGeometry &geometry,
    RawStats &stats)
{
    stats.input += left_points.size();
    std::vector<cv::Point2f> right_points, reverse_points;
    std::vector<unsigned char> forward_status, reverse_status;
    runKlt(
        left, right, left_points, right_points, reverse_points,
        forward_status, reverse_status);
    for (int index = 0; index < static_cast<int>(left_points.size()); ++index)
    {
        if (!forward_status[index] ||
            !inBorder(right_points[index], right.cols, right.rows))
            continue;
        ++stats.forward;
        if (index >= static_cast<int>(reverse_status.size()) ||
            !reverse_status[index])
            continue;
        ++stats.reverse;
        const double fb = cv::norm(left_points[index] - reverse_points[index]);
        if (fb <= 1.5)
            ++stats.fb15;
        if (fb <= 1.0)
            ++stats.fb10;
        if (fb > 0.5)
            continue;
        ++stats.fb05;

        const double epipolar_error =
            geometry.epipolarPixelError(left_points[index], right_points[index]);
        if (epipolar_error <= 3.0)
            ++stats.epipolar30;
        if (epipolar_error > 1.5)
            continue;
        ++stats.epipolar15;

        const float disparity =
            left_points[index].x - right_points[index].x;
        if (disparity >= 0.0f && disparity <= 160.0f)
            ++stats.horizontal_disparity;
        const bool positive = geometry.hasPositiveDepth(
            left_points[index], right_points[index]);
        if (positive)
        {
            ++stats.positive_depth;
            if (disparity < 0.0f)
                ++stats.negative_disparity_positive_depth;
        }
    }
}

void analyzeRectified(
    const cv::Mat &rectified_left,
    const cv::Mat &rectified_right,
    const std::vector<cv::Point2f> &raw_left_points,
    const StereoGeometry &geometry,
    RectifiedStats &stats)
{
    stats.input += raw_left_points.size();
    std::vector<cv::Point2f> left_points;
    for (const cv::Point2f &raw : raw_left_points)
    {
        cv::Point2f rectified;
        if (geometry.rawLeftToRectified(raw, rectified))
            left_points.push_back(rectified);
    }
    stats.mapped += left_points.size();
    std::vector<cv::Point2f> right_points, reverse_points;
    std::vector<unsigned char> forward_status, reverse_status;
    runKlt(
        rectified_left, rectified_right, left_points,
        right_points, reverse_points, forward_status, reverse_status);
    for (int index = 0; index < static_cast<int>(left_points.size()); ++index)
    {
        if (!forward_status[index] ||
            !inBorder(
                right_points[index],
                rectified_right.cols, rectified_right.rows))
            continue;
        ++stats.forward;
        if (!reverse_status[index])
            continue;
        ++stats.reverse;
        const double fb = cv::norm(left_points[index] - reverse_points[index]);
        if (fb <= 1.0)
            ++stats.fb10;
        if (fb > 0.5)
            continue;
        ++stats.fb05;
        if (std::abs(left_points[index].y - right_points[index].y) <= 1.5)
        {
            ++stats.row15;
            const float disparity =
                left_points[index].x - right_points[index].x;
            if (disparity >= 0.0f && disparity <= 160.0f)
                ++stats.positive_disparity;
        }
    }
}

void printRaw(const std::string &name, const RawStats &stats)
{
    std::cout << "\nRAW " << name << "\n"
              << "  input:              " << stats.input << "\n"
              << "  forward:            " << stats.forward << " ("
              << percentage(stats.forward, stats.input) << "%)\n"
              << "  reverse:            " << stats.reverse << " ("
              << percentage(stats.reverse, stats.input) << "%)\n"
              << "  FB <= 0.5:          " << stats.fb05 << " ("
              << percentage(stats.fb05, stats.input) << "%)\n"
              << "  FB <= 1.0:          " << stats.fb10 << " ("
              << percentage(stats.fb10, stats.input) << "%)\n"
              << "  FB <= 1.5:          " << stats.fb15 << " ("
              << percentage(stats.fb15, stats.input) << "%)\n"
              << "  FB0.5 + epi1.5:     " << stats.epipolar15 << " ("
              << percentage(stats.epipolar15, stats.input) << "%)\n"
              << "  FB0.5 + epi3.0:     " << stats.epipolar30 << " ("
              << percentage(stats.epipolar30, stats.input) << "%)\n"
              << "  production disparity:" << stats.horizontal_disparity << " ("
              << percentage(stats.horizontal_disparity, stats.input) << "%)\n"
              << "  positive depth:      " << stats.positive_depth << " ("
              << percentage(stats.positive_depth, stats.input) << "%)\n"
              << "  positive depth with negative raw disparity: "
              << stats.negative_disparity_positive_depth << "\n";
}

void printRectified(const std::string &name, const RectifiedStats &stats)
{
    std::cout << "\nRECTIFIED " << name << "\n"
              << "  input:              " << stats.input << "\n"
              << "  mapped in bounds:   " << stats.mapped << " ("
              << percentage(stats.mapped, stats.input) << "%)\n"
              << "  forward:            " << stats.forward << " ("
              << percentage(stats.forward, stats.mapped) << "% mapped)\n"
              << "  reverse:            " << stats.reverse << " ("
              << percentage(stats.reverse, stats.mapped) << "% mapped)\n"
              << "  FB <= 0.5:          " << stats.fb05 << " ("
              << percentage(stats.fb05, stats.mapped) << "% mapped)\n"
              << "  FB <= 1.0:          " << stats.fb10 << " ("
              << percentage(stats.fb10, stats.mapped) << "% mapped)\n"
              << "  FB0.5 + row1.5:     " << stats.row15 << " ("
              << percentage(stats.row15, stats.mapped) << "% mapped)\n"
              << "  + disparity 0..160: " << stats.positive_disparity << " ("
              << percentage(stats.positive_disparity, stats.mapped)
              << "% mapped)\n";
}

void analyzeTemporalConsistency(
    const cv::Mat &previous_image,
    const cv::Mat &current_image,
    const std::map<int, cv::Point2f> &previous_points,
    const std::map<int, cv::Point2f> &current_points,
    TemporalConsistencyStats &stats,
    std::map<int, cv::Point2f> &current_klt_points)
{
    current_klt_points.clear();
    if (previous_image.empty() || previous_points.empty())
        return;
    ++stats.transitions;
    stats.previous_points += previous_points.size();

    std::vector<int> ids;
    std::vector<cv::Point2f> points;
    ids.reserve(previous_points.size());
    points.reserve(previous_points.size());
    for (const auto &item : previous_points)
    {
        ids.push_back(item.first);
        points.push_back(item.second);
        if (current_points.count(item.first))
            ++stats.lightglue_inherited;
    }

    std::vector<cv::Point2f> tracked, reverse;
    std::vector<unsigned char> forward_status, reverse_status;
    runKlt(
        previous_image, current_image, points,
        tracked, reverse, forward_status, reverse_status);
    for (int index = 0; index < static_cast<int>(ids.size()); ++index)
    {
        if (!forward_status[index] || !reverse_status[index] ||
            !inBorder(tracked[index], current_image.cols, current_image.rows) ||
            cv::norm(points[index] - reverse[index]) > 0.5)
            continue;
        ++stats.klt_fb_valid;
        const auto current = current_points.find(ids[index]);
        if (current == current_points.end())
            continue;
        ++stats.inherited_with_klt_fb;
        current_klt_points[ids[index]] = tracked[index];
        const double error = cv::norm(tracked[index] - current->second);
        stats.coordinate_errors.push_back(error);
        if (error <= 0.5)
            ++stats.agreement05;
        if (error <= 1.0)
            ++stats.agreement10;
        if (error <= 2.0)
            ++stats.agreement20;
        if (error <= 3.0)
            ++stats.agreement30;
        if (error <= 5.0)
            ++stats.agreement50;
    }
}

long countBelow(
    const std::vector<double> &values, double threshold)
{
    return std::count_if(
        values.begin(), values.end(),
        [threshold](double value) { return value < threshold; });
}

long countAbove(
    const std::vector<double> &values, double threshold)
{
    return std::count_if(
        values.begin(), values.end(),
        [threshold](double value) { return value > threshold; });
}

double percentile(std::vector<double> values, double fraction);

void analyzeStereoDepth(
    const std::map<int, std::vector<std::pair<
        int, Eigen::Matrix<double, 7, 1>>>> &observations,
    const std::map<int, cv::Point2f> &previous_points,
    const std::map<int, cv::Point2f> &temporal_klt_points,
    const StereoGeometry &geometry,
    StereoDepthStats &stats)
{
    for (const auto &feature : observations)
    {
        bool has_left = false;
        bool has_right = false;
        cv::Point2f left, right;
        for (const auto &camera_observation : feature.second)
        {
            const auto &measurement = camera_observation.second;
            const cv::Point2f point(
                static_cast<float>(measurement(3)),
                static_cast<float>(measurement(4)));
            if (camera_observation.first == 0)
            {
                left = point;
                has_left = true;
            }
            else if (camera_observation.first == 1)
            {
                right = point;
                has_right = true;
            }
        }
        if (!has_left || !has_right)
            continue;
        ++stats.stereo_observations;
        if (previous_points.count(feature.first))
            ++stats.inherited_observations;

        double left_depth = 0.0;
        double right_depth = 0.0;
        double parallax = 0.0;
        if (!geometry.triangulate(
                left, right, left_depth, right_depth, parallax) ||
            left_depth <= 0.0 || right_depth <= 0.0)
            continue;
        stats.depths.push_back(left_depth);
        stats.parallax_degrees.push_back(parallax);
        cv::Point2f rectified_left, rectified_right;
        if (!geometry.rawLeftToRectified(left, rectified_left) ||
            !geometry.rawRightToRectified(right, rectified_right))
            continue;
        const double disparity = rectified_left.x - rectified_right.x;
        stats.rectified_disparities.push_back(disparity);

        const auto klt = temporal_klt_points.find(feature.first);
        if (klt == temporal_klt_points.end())
            continue;
        double klt_left_depth = 0.0;
        double klt_right_depth = 0.0;
        double klt_parallax = 0.0;
        cv::Point2f rectified_klt_left;
        if (!geometry.triangulate(
                klt->second, right, klt_left_depth,
                klt_right_depth, klt_parallax) ||
            klt_left_depth <= 0.0 || klt_right_depth <= 0.0 ||
            !geometry.rawLeftToRectified(
                klt->second, rectified_klt_left))
            continue;
        ++stats.temporal_klt_comparable;
        stats.temporal_left_shifts.push_back(
            cv::norm(left - klt->second));
        stats.temporal_disparity_changes.push_back(std::abs(
            (rectified_klt_left.x - rectified_right.x) - disparity));
        stats.temporal_relative_depth_changes.push_back(
            std::abs(klt_left_depth - left_depth) /
            std::max(left_depth, 1e-9));
    }
}

void printStereoDepth(const StereoDepthStats &stats)
{
    std::cout << "\nSTEREO DEPTH SENSITIVITY\n"
              << "  stereo observations:      "
              << stats.stereo_observations << "\n"
              << "  inherited observations:   "
              << stats.inherited_observations << " ("
              << percentage(
                     stats.inherited_observations,
                     stats.stereo_observations)
              << "%)\n"
              << "  depth p50/p90/p95/p99 m:   "
              << percentile(stats.depths, 0.50) << " / "
              << percentile(stats.depths, 0.90) << " / "
              << percentile(stats.depths, 0.95) << " / "
              << percentile(stats.depths, 0.99) << "\n"
              << "  depth >20/50/100 m:        "
              << percentage(countAbove(stats.depths, 20.0), stats.depths.size())
              << "% / "
              << percentage(countAbove(stats.depths, 50.0), stats.depths.size())
              << "% / "
              << percentage(countAbove(stats.depths, 100.0), stats.depths.size())
              << "%\n"
              << "  disparity p01/p05/p10/p50 px: "
              << percentile(stats.rectified_disparities, 0.01) << " / "
              << percentile(stats.rectified_disparities, 0.05) << " / "
              << percentile(stats.rectified_disparities, 0.10) << " / "
              << percentile(stats.rectified_disparities, 0.50) << "\n"
              << "  disparity <1/2/3/5 px:     "
              << percentage(
                     countBelow(stats.rectified_disparities, 1.0),
                     stats.rectified_disparities.size())
              << "% / "
              << percentage(
                     countBelow(stats.rectified_disparities, 2.0),
                     stats.rectified_disparities.size())
              << "% / "
              << percentage(
                     countBelow(stats.rectified_disparities, 3.0),
                     stats.rectified_disparities.size())
              << "% / "
              << percentage(
                     countBelow(stats.rectified_disparities, 5.0),
                     stats.rectified_disparities.size())
              << "%\n"
              << "  parallax p01/p05/p10/p50 deg: "
              << percentile(stats.parallax_degrees, 0.01) << " / "
              << percentile(stats.parallax_degrees, 0.05) << " / "
              << percentile(stats.parallax_degrees, 0.10) << " / "
              << percentile(stats.parallax_degrees, 0.50) << "\n"
              << "  temporal-KLT comparable:   "
              << stats.temporal_klt_comparable << "\n"
              << "  left shift p50/p90/p95/p99 px: "
              << percentile(stats.temporal_left_shifts, 0.50) << " / "
              << percentile(stats.temporal_left_shifts, 0.90) << " / "
              << percentile(stats.temporal_left_shifts, 0.95) << " / "
              << percentile(stats.temporal_left_shifts, 0.99) << "\n"
              << "  disparity change p50/p90/p95/p99 px: "
              << percentile(stats.temporal_disparity_changes, 0.50) << " / "
              << percentile(stats.temporal_disparity_changes, 0.90) << " / "
              << percentile(stats.temporal_disparity_changes, 0.95) << " / "
              << percentile(stats.temporal_disparity_changes, 0.99) << "\n"
              << "  relative depth change p50/p90/p95/p99: "
              << percentage(
                     static_cast<long>(std::round(
                         percentile(
                             stats.temporal_relative_depth_changes, 0.50) *
                         10000.0)),
                     10000)
              << "% / "
              << percentage(
                     static_cast<long>(std::round(
                         percentile(
                             stats.temporal_relative_depth_changes, 0.90) *
                         10000.0)),
                     10000)
              << "% / "
              << percentage(
                     static_cast<long>(std::round(
                         percentile(
                             stats.temporal_relative_depth_changes, 0.95) *
                         10000.0)),
                     10000)
              << "% / "
              << percentage(
                     static_cast<long>(std::round(
                         percentile(
                             stats.temporal_relative_depth_changes, 0.99) *
                         10000.0)),
                     10000)
              << "%\n";
}

double percentile(std::vector<double> values, double fraction)
{
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
        std::round(fraction * static_cast<double>(values.size() - 1)));
    return values[index];
}

void printTemporal(const TemporalConsistencyStats &stats)
{
    double sum = 0.0;
    double squared_sum = 0.0;
    for (double error : stats.coordinate_errors)
    {
        sum += error;
        squared_sum += error * error;
    }
    const double count = static_cast<double>(stats.coordinate_errors.size());
    std::cout << "\nTEMPORAL LIGHTGLUE ID VS LEFT KLT\n"
              << "  consecutive transitions: " << stats.transitions << "\n"
              << "  previous selected points: " << stats.previous_points << "\n"
              << "  LightGlue inherited IDs:  " << stats.lightglue_inherited
              << " (" << percentage(
                     stats.lightglue_inherited, stats.previous_points)
              << "% previous)\n"
              << "  temporal KLT FB valid:     " << stats.klt_fb_valid
              << " (" << percentage(stats.klt_fb_valid, stats.previous_points)
              << "% previous)\n"
              << "  inherited + KLT valid:     "
              << stats.inherited_with_klt_fb << "\n"
              << "  coordinate agreement <=0.5/1/2/3/5 px: "
              << stats.agreement05 << " / " << stats.agreement10 << " / "
              << stats.agreement20 << " / " << stats.agreement30 << " / "
              << stats.agreement50 << "\n"
              << "  agreement percentages: "
              << percentage(stats.agreement05, stats.inherited_with_klt_fb)
              << "% / "
              << percentage(stats.agreement10, stats.inherited_with_klt_fb)
              << "% / "
              << percentage(stats.agreement20, stats.inherited_with_klt_fb)
              << "% / "
              << percentage(stats.agreement30, stats.inherited_with_klt_fb)
              << "% / "
              << percentage(stats.agreement50, stats.inherited_with_klt_fb)
              << "%\n"
              << "  coordinate error mean/rmse/p50/p90/p95/p99 px: "
              << (count > 0.0 ? sum / count : 0.0) << " / "
              << (count > 0.0 ? std::sqrt(squared_sum / count) : 0.0) << " / "
              << percentile(stats.coordinate_errors, 0.50) << " / "
              << percentile(stats.coordinate_errors, 0.90) << " / "
              << percentile(stats.coordinate_errors, 0.95) << " / "
              << percentile(stats.coordinate_errors, 0.99) << "\n";
}

void analyzeStereoCorrespondence(
    const cv::Mat &left,
    const cv::Mat &right,
    const cv::Mat &rectified_left,
    const cv::Mat &rectified_right,
    const std::vector<cv::Point2f> &raw_left_points,
    const StereoGeometry &geometry,
    StereoCorrespondenceStats &stats)
{
    stats.input += raw_left_points.size();
    std::vector<cv::Point2f> raw_right, raw_reverse;
    std::vector<unsigned char> raw_forward_status, raw_reverse_status;
    runKlt(
        left, right, raw_left_points,
        raw_right, raw_reverse, raw_forward_status, raw_reverse_status);

    std::vector<cv::Point2f> rectified_left_points;
    std::vector<int> raw_indices;
    for (int index = 0;
         index < static_cast<int>(raw_left_points.size()); ++index)
    {
        cv::Point2f rectified;
        if (geometry.rawLeftToRectified(raw_left_points[index], rectified))
        {
            rectified_left_points.push_back(rectified);
            raw_indices.push_back(index);
        }
    }
    std::vector<cv::Point2f> rectified_right_points, rectified_reverse;
    std::vector<unsigned char> rectified_forward_status;
    std::vector<unsigned char> rectified_reverse_status;
    runKlt(
        rectified_left, rectified_right, rectified_left_points,
        rectified_right_points, rectified_reverse,
        rectified_forward_status, rectified_reverse_status);

    for (int mapped_index = 0;
         mapped_index < static_cast<int>(raw_indices.size()); ++mapped_index)
    {
        const int raw_index = raw_indices[mapped_index];
        const bool raw_valid =
            raw_forward_status[raw_index] &&
            raw_reverse_status[raw_index] &&
            inBorder(raw_right[raw_index], right.cols, right.rows) &&
            cv::norm(
                raw_left_points[raw_index] - raw_reverse[raw_index]) <= 0.5;
        if (raw_valid)
            ++stats.raw_fb_valid;

        bool rectified_valid =
            rectified_forward_status[mapped_index] &&
            rectified_reverse_status[mapped_index] &&
            inBorder(
                rectified_right_points[mapped_index],
                rectified_right.cols, rectified_right.rows) &&
            cv::norm(
                rectified_left_points[mapped_index] -
                rectified_reverse[mapped_index]) <= 0.5 &&
            std::abs(
                rectified_left_points[mapped_index].y -
                rectified_right_points[mapped_index].y) <= 1.5;
        cv::Point2f rectified_reference_raw;
        if (rectified_valid)
        {
            rectified_valid = geometry.rectifiedRightToRaw(
                rectified_right_points[mapped_index], rectified_reference_raw);
        }
        if (rectified_valid)
            ++stats.rectified_fb_valid;
        if (!raw_valid || !rectified_valid)
            continue;

        ++stats.both_valid;
        const double error =
            cv::norm(raw_right[raw_index] - rectified_reference_raw);
        stats.right_pixel_errors.push_back(error);
        if (error <= 0.5)
            ++stats.agreement05;
        if (error <= 1.0)
            ++stats.agreement10;
        if (error <= 2.0)
            ++stats.agreement20;
        if (error <= 3.0)
            ++stats.agreement30;
        if (error <= 5.0)
            ++stats.agreement50;
    }
}

void printStereoCorrespondence(const StereoCorrespondenceStats &stats)
{
    double sum = 0.0;
    double squared_sum = 0.0;
    for (double error : stats.right_pixel_errors)
    {
        sum += error;
        squared_sum += error * error;
    }
    const double count = static_cast<double>(stats.right_pixel_errors.size());
    std::cout << "\nRAW STEREO KLT VS RECTIFIED KLT REFERENCE\n"
              << "  input points:             " << stats.input << "\n"
              << "  raw FB valid (mapped):    " << stats.raw_fb_valid << "\n"
              << "  rectified FB+row valid:   " << stats.rectified_fb_valid << "\n"
              << "  valid in both paths:      " << stats.both_valid << "\n"
              << "  right agreement <=0.5/1/2/3/5 px: "
              << stats.agreement05 << " / " << stats.agreement10 << " / "
              << stats.agreement20 << " / " << stats.agreement30 << " / "
              << stats.agreement50 << "\n"
              << "  agreement percentages: "
              << percentage(stats.agreement05, stats.both_valid) << "% / "
              << percentage(stats.agreement10, stats.both_valid) << "% / "
              << percentage(stats.agreement20, stats.both_valid) << "% / "
              << percentage(stats.agreement30, stats.both_valid) << "% / "
              << percentage(stats.agreement50, stats.both_valid) << "%\n"
              << "  right error mean/rmse/p50/p90/p95/p99 px: "
              << (count > 0.0 ? sum / count : 0.0) << " / "
              << (count > 0.0 ? std::sqrt(squared_sum / count) : 0.0) << " / "
              << percentile(stats.right_pixel_errors, 0.50) << " / "
              << percentile(stats.right_pixel_errors, 0.90) << " / "
              << percentile(stats.right_pixel_errors, 0.95) << " / "
              << percentile(stats.right_pixel_errors, 0.99) << "\n";
}
}  // namespace

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: " << argv[0]
                  << " CONFIG IMAGE_DIRECTORY FRAME_COUNT\n";
        return 2;
    }
    rclcpp::init(argc, argv);
    readParameters(argv[1]);
    YOLOPointLightGlueFeatureTracker tracker;
    tracker.readIntrinsicParameter(CAM_NAMES);
    const auto camera0 =
        camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
            CAM_NAMES[0]);
    const auto camera1 =
        camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
            CAM_NAMES[1]);

    const std::string directory = argv[2];
    const int frame_count = std::atoi(argv[3]);
    RawStats yolo_raw, gftt_raw;
    RectifiedStats yolo_rectified, gftt_raw_rectified, gftt_rectified;
    TemporalConsistencyStats temporal_stats;
    StereoCorrespondenceStats stereo_correspondence_stats;
    StereoDepthStats stereo_depth_stats;
    std::vector<double> yolo_corner_eigenvalues;
    std::vector<double> stereo_corner_eigenvalues;
    std::vector<double> gftt_corner_eigenvalues;
    std::unique_ptr<StereoGeometry> geometry;
    std::vector<int> source_indices(frame_count);
    std::iota(source_indices.begin(), source_indices.end(), 0);
    {
        std::ifstream mapping(directory + "/source_indices.txt");
        int output_index = 0;
        int source_index = 0;
        while (mapping >> output_index >> source_index)
        {
            if (output_index >= 0 && output_index < frame_count)
                source_indices[output_index] = source_index;
        }
    }
    cv::Mat previous_left;
    std::map<int, cv::Point2f> previous_yolo_points;

    for (int frame = 0; frame < frame_count; ++frame)
    {
        char left_name[1024], right_name[1024];
        std::snprintf(
            left_name, sizeof(left_name), "%s/left_%03d.png",
            directory.c_str(), frame);
        std::snprintf(
            right_name, sizeof(right_name), "%s/right_%03d.png",
            directory.c_str(), frame);
        const cv::Mat left = cv::imread(left_name, cv::IMREAD_GRAYSCALE);
        const cv::Mat right = cv::imread(right_name, cv::IMREAD_GRAYSCALE);
        if (left.empty() || right.empty())
        {
            std::cerr << "failed to read pair " << frame << "\n";
            return 3;
        }
        if (!geometry)
            geometry.reset(new StereoGeometry(
                camera0, camera1, left.cols, left.rows));

        const auto observations = tracker.trackImage(
            1403636578.0 + frame * 0.05, left, right);
        std::vector<cv::Point2f> yolo_points;
        std::map<int, cv::Point2f> yolo_points_by_id;
        std::set<int> stereo_ids;
        std::map<int, cv::Point2f> temporal_klt_points;
        for (const auto &feature : observations)
        {
            for (const auto &camera_observation : feature.second)
            {
                if (camera_observation.first == 0)
                {
                    const auto &measurement = camera_observation.second;
                    yolo_points.emplace_back(
                        static_cast<float>(measurement(3)),
                        static_cast<float>(measurement(4)));
                    yolo_points_by_id[feature.first] = yolo_points.back();
                }
                else if (camera_observation.first == 1)
                {
                    stereo_ids.insert(feature.first);
                }
            }
        }

        if (frame > 0 &&
            source_indices[frame] == source_indices[frame - 1] + 1)
        {
            analyzeTemporalConsistency(
                previous_left, left,
                previous_yolo_points, yolo_points_by_id,
                temporal_stats, temporal_klt_points);
        }
        analyzeStereoDepth(
            observations, previous_yolo_points, temporal_klt_points,
            *geometry, stereo_depth_stats);
        previous_left = left;
        previous_yolo_points.swap(yolo_points_by_id);

        std::vector<cv::Point2f> gftt_points;
        cv::goodFeaturesToTrack(
            left, gftt_points,
            std::max(1, static_cast<int>(yolo_points.size())),
            0.01, MIN_DIST);
        cv::Mat corner_eigenvalues;
        cv::cornerMinEigenVal(left, corner_eigenvalues, 7, 3);
        for (const auto &item : yolo_points_by_id)
        {
            const cv::Point point(
                cvRound(item.second.x), cvRound(item.second.y));
            if (point.x < 0 || point.y < 0 ||
                point.x >= corner_eigenvalues.cols ||
                point.y >= corner_eigenvalues.rows)
                continue;
            const double value =
                corner_eigenvalues.at<float>(point);
            yolo_corner_eigenvalues.push_back(value);
            if (stereo_ids.count(item.first))
                stereo_corner_eigenvalues.push_back(value);
        }
        for (const cv::Point2f &point2f : gftt_points)
        {
            const cv::Point point(
                cvRound(point2f.x), cvRound(point2f.y));
            if (point.x >= 0 && point.y >= 0 &&
                point.x < corner_eigenvalues.cols &&
                point.y < corner_eigenvalues.rows)
                gftt_corner_eigenvalues.push_back(
                    corner_eigenvalues.at<float>(point));
        }

        cv::Mat rectified_left, rectified_right;
        geometry->rectify(
            left, right, rectified_left, rectified_right);
        analyzeStereoCorrespondence(
            left, right, rectified_left, rectified_right,
            yolo_points, *geometry, stereo_correspondence_stats);
        std::vector<cv::Point2f> rectified_gftt_points;
        cv::goodFeaturesToTrack(
            rectified_left, rectified_gftt_points,
            std::max(1, static_cast<int>(yolo_points.size())),
            0.01, MIN_DIST);

        analyzeRaw(
            left, right, yolo_points, *geometry, yolo_raw);
        analyzeRaw(
            left, right, gftt_points, *geometry, gftt_raw);
        analyzeRectified(
            rectified_left, rectified_right,
            yolo_points, *geometry, yolo_rectified);
        analyzeRectified(
            rectified_left, rectified_right,
            gftt_points, *geometry, gftt_raw_rectified);

        // Rectified GFTT points are already in the rectified coordinate system.
        RectifiedStats frame_rectified;
        frame_rectified.input = rectified_gftt_points.size();
        frame_rectified.mapped = rectified_gftt_points.size();
        std::vector<cv::Point2f> right_points, reverse_points;
        std::vector<unsigned char> forward_status, reverse_status;
        runKlt(
            rectified_left, rectified_right, rectified_gftt_points,
            right_points, reverse_points, forward_status, reverse_status);
        for (int index = 0;
             index < static_cast<int>(rectified_gftt_points.size()); ++index)
        {
            if (!forward_status[index] ||
                !inBorder(right_points[index], right.cols, right.rows))
                continue;
            ++frame_rectified.forward;
            if (!reverse_status[index])
                continue;
            ++frame_rectified.reverse;
            const double fb =
                cv::norm(rectified_gftt_points[index] - reverse_points[index]);
            if (fb <= 1.0)
                ++frame_rectified.fb10;
            if (fb > 0.5)
                continue;
            ++frame_rectified.fb05;
            if (std::abs(
                    rectified_gftt_points[index].y -
                    right_points[index].y) <= 1.5)
            {
                ++frame_rectified.row15;
                const float disparity =
                    rectified_gftt_points[index].x - right_points[index].x;
                if (disparity >= 0.0f && disparity <= 160.0f)
                    ++frame_rectified.positive_disparity;
            }
        }
#define ACCUMULATE(member) \
        gftt_rectified.member += frame_rectified.member
        ACCUMULATE(input);
        ACCUMULATE(mapped);
        ACCUMULATE(forward);
        ACCUMULATE(reverse);
        ACCUMULATE(fb05);
        ACCUMULATE(fb10);
        ACCUMULATE(row15);
        ACCUMULATE(positive_disparity);
#undef ACCUMULATE

        if ((frame + 1) % 20 == 0)
            std::cout << "processed " << frame + 1 << "/"
                      << frame_count << " pairs\n";
    }

    std::cout << std::fixed << std::setprecision(2);
    printRaw("YOLOPOINT_SELECTED", yolo_raw);
    printRaw("GFTT_RAW", gftt_raw);
    printRectified("YOLOPOINT_MAPPED", yolo_rectified);
    printRectified("GFTT_RAW_MAPPED", gftt_raw_rectified);
    printRectified("GFTT_ON_RECTIFIED_IMAGE", gftt_rectified);
    printTemporal(temporal_stats);
    printStereoCorrespondence(stereo_correspondence_stats);
    printStereoDepth(stereo_depth_stats);
    std::cout << std::scientific << std::setprecision(6)
              << "\nLEFT CORNER MIN-EIGENVALUE (7x7)\n"
              << "  YOLO p01/p05/p10/p50: "
              << percentile(yolo_corner_eigenvalues, 0.01) << " / "
              << percentile(yolo_corner_eigenvalues, 0.05) << " / "
              << percentile(yolo_corner_eigenvalues, 0.10) << " / "
              << percentile(yolo_corner_eigenvalues, 0.50) << "\n"
              << "  YOLO stereo p01/p05/p10/p50: "
              << percentile(stereo_corner_eigenvalues, 0.01) << " / "
              << percentile(stereo_corner_eigenvalues, 0.05) << " / "
              << percentile(stereo_corner_eigenvalues, 0.10) << " / "
              << percentile(stereo_corner_eigenvalues, 0.50) << "\n"
              << "  GFTT p01/p05/p10/p50: "
              << percentile(gftt_corner_eigenvalues, 0.01) << " / "
              << percentile(gftt_corner_eigenvalues, 0.05) << " / "
              << percentile(gftt_corner_eigenvalues, 0.10) << " / "
              << percentile(gftt_corner_eigenvalues, 0.50) << "\n"
              << std::fixed << std::setprecision(2);
    rclcpp::shutdown();
    return 0;
}

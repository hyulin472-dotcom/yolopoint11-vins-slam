/*******************************************************
 * Online keyframe dense stereo mapper for RViz visualization.
 *******************************************************/

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

class OnlineDenseMapper
{
  public:
    void init(rclcpp::Node::SharedPtr node);
    void reset();
    void shutdown();
    void processKeyframe(const cv::Mat &left,
                         const cv::Mat &right,
                         const cv::Mat &left_color,
                         const Eigen::Vector3d &p_wb,
                         const Eigen::Matrix3d &r_wb,
                         const std_msgs::msg::Header &header);

  private:
    bool initialized_ = false;
    int keyframe_counter_ = 0;
    double fx_ = 460.0;
    double cx_ = 0.0;
    double cy_ = 0.0;
    double baseline_ = 0.0;
    cv::Mat map00_, map01_, map10_, map11_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    std::vector<Eigen::Vector3f> points_;
    std::vector<uint32_t> colors_;
    std::unordered_set<int64_t> occupied_voxels_;
    struct DenseSubmap
    {
        int64_t stamp_key = 0;
        Eigen::Vector3d p_wb;
        Eigen::Matrix3d r_wb;
        std::vector<Eigen::Vector3f> local_points;
        std::vector<uint32_t> colors;
    };
    std::vector<DenseSubmap> submaps_;
    std::unordered_map<int64_t, size_t> submap_index_by_stamp_;
    Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "OnlineDenseFoundationStereo"};
    Ort::SessionOptions ort_session_options_;
    std::unique_ptr<Ort::Session> ort_session_;
    Ort::AllocatorWithDefaultOptions ort_allocator_;
    std::vector<std::string> input_name_storage_;
    std::vector<std::string> output_name_storage_;
    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;
    std::vector<int64_t> input_shape_{1, 3, 320, 736};

    bool buildRectifier();
    bool initFoundationStereo();
    bool inferFoundationStereo(const cv::Mat &left_rect, const cv::Mat &right_rect, cv::Mat &disparity);
    std::vector<float> imageToTensor(const cv::Mat &image) const;
    void publish(const std_msgs::msg::Header &header);
    void publishSubmaps(const std_msgs::msg::Header &header);
    int64_t voxelKey(const Eigen::Vector3f &p) const;
    int64_t stampKey(const std_msgs::msg::Header &header) const;
};

#pragma once

#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

class YOLOPointCudaPostprocessor
{
  public:
    YOLOPointCudaPostprocessor(
        int image_width,
        int image_height,
        int descriptor_dimension,
        int max_keypoints);
    ~YOLOPointCudaPostprocessor();

    YOLOPointCudaPostprocessor(const YOLOPointCudaPostprocessor &) = delete;
    YOLOPointCudaPostprocessor &operator=(
        const YOLOPointCudaPostprocessor &) = delete;

    float *semiDeviceData();
    float *denseDescriptorDeviceData();
    size_t semiElementCount() const;
    size_t denseDescriptorElementCount() const;
    void globalDescriptor(cv::Mat &descriptor);

    void process(
        float score_threshold,
        int nms_radius,
        int remove_borders,
        float scale_x,
        float scale_y,
        std::vector<cv::KeyPoint> &keypoints,
        cv::Mat &descriptors);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

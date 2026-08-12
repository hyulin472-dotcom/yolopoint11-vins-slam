#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <cmath>

#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>

#include "../vins/src/estimator/parameters.h"
#include "../vins/src/featureTracker/yolopoint_lightglue_feature_tracker.h"
#include "../vins/src/loop_closure/independent_loop_closure.h"

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: " << argv[0]
                  << " CONFIG IMAGE_DIRECTORY FRAME_COUNT" << std::endl;
        return 2;
    }

    rclcpp::init(argc, argv);
    readParameters(argv[1]);
    if (FEATURE_TRACKER_TYPE != 3)
    {
        std::cerr << "smoke test requires feature_tracker_type: 3" << std::endl;
        return 2;
    }

    YOLOPointLightGlueFeatureTracker tracker;
    tracker.readIntrinsicParameter(CAM_NAMES);
    const std::string directory = argv[2];
    const int frame_count = std::atoi(argv[3]);
    std::set<int> previous_ids;
    int total_temporal = 0;
    int total_stereo = 0;

    for (int frame = 0; frame < frame_count; ++frame)
    {
        char left_name[1024];
        char right_name[1024];
        std::snprintf(left_name, sizeof(left_name), "%s/left_%03d.png",
                      directory.c_str(), frame);
        std::snprintf(right_name, sizeof(right_name), "%s/right_%03d.png",
                      directory.c_str(), frame);
        const cv::Mat left = cv::imread(left_name, cv::IMREAD_GRAYSCALE);
        const cv::Mat right = cv::imread(right_name, cv::IMREAD_GRAYSCALE);
        if (left.empty() || right.empty())
        {
            std::cerr << "failed to read pair " << frame << std::endl;
            return 3;
        }

        const auto observations = tracker.trackImage(
            1403636578.0 + frame * 0.05, left, right);
        if (LOOP_CLOSURE_ENABLE && LOOP_CLOSURE_MODE == 3)
        {
            const LoopModelFeatures loop_features = tracker.getLoopFeatures();
            const double global_norm = cv::norm(loop_features.global_descriptor,
                                                cv::NORM_L2);
            if (!loop_features.valid() ||
                loop_features.keypoints.size() != observations.size() ||
                std::abs(global_norm - 1.0) > 1e-3)
            {
                std::cerr << "invalid YOLOPoint loop feature snapshot at frame "
                          << frame << ", norm=" << global_norm << std::endl;
                return 7;
            }
        }
        std::set<int> current_ids;
        int temporal = 0;
        int stereo = 0;
        for (const auto &feature : observations)
        {
            current_ids.insert(feature.first);
            temporal += previous_ids.count(feature.first) ? 1 : 0;
            stereo += feature.second.size() > 1 ? 1 : 0;
        }
        total_temporal += temporal;
        total_stereo += stereo;
        std::cout << "frame=" << frame
                  << " left=" << observations.size()
                  << " inherited=" << temporal
                  << " stereo=" << stereo << std::endl;
        if (observations.empty())
            return 4;
        previous_ids.swap(current_ids);
    }
    if (frame_count > 1 && total_temporal == 0)
    {
        std::cerr << "no temporal LightGlue IDs were inherited" << std::endl;
        return 5;
    }
    if (total_stereo == 0)
    {
        std::cerr << "no stereo KLT observations survived" << std::endl;
        return 6;
    }
    std::cout << "smoke_test_passed temporal=" << total_temporal
              << " stereo=" << total_stereo << std::endl;
    if (LOOP_CLOSURE_ENABLE)
    {
        auto node = rclcpp::Node::make_shared("loop_mode_smoke");
        independentLoopClosure().init(node);
        if (!independentLoopClosure().enabled())
        {
            std::cerr << "loop backend failed to initialize" << std::endl;
            return 8;
        }
        independentLoopClosure().reset();
        std::cout << "loop_backend_initialized mode=" << LOOP_CLOSURE_MODE
                  << std::endl;
    }
    rclcpp::shutdown();
    return 0;
}

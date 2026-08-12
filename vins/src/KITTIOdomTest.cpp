/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <cmath>
#include <cstdlib>
#include <set>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include "estimator/estimator.h"
#include "utility/visualization.h"
#include "loop_closure/independent_loop_closure.h"

using namespace std;
using namespace Eigen;

Estimator estimator;

Eigen::Matrix3d c1Rc0, c0Rc1;
Eigen::Vector3d c1Tc0, c0Tc1;

namespace
{
std::set<size_t> parseExportFrames(const char *value)
{
	std::set<size_t> frames;
	if (value == nullptr || *value == '\0')
		return frames;

	std::stringstream stream(value);
	std::string token;
	while (std::getline(stream, token, ','))
	{
		try
		{
			size_t parsed = 0;
			const unsigned long frame = std::stoul(token, &parsed);
			if (parsed == token.size())
				frames.insert(static_cast<size_t>(frame));
		}
		catch (const std::exception &)
		{
			std::cerr << "Ignore invalid KITTI_EXPORT_TRACK_FRAMES entry: "
			          << token << std::endl;
		}
	}
	return frames;
}
}

int main(int argc, char** argv)
{
	rclcpp::init(argc, argv);
	auto n = rclcpp::Node::make_shared("vins_estimator");
	// ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pubLeftImage  = n->create_publisher<sensor_msgs::msg::Image>("/leftImage",1000);
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pubRightImage = n->create_publisher<sensor_msgs::msg::Image>("/rightImage",1000);

	if(argc != 3)
	{
		printf("please intput: rosrun vins kitti_odom_test [config file] [data folder] \n"
			   "for example: rosrun vins kitti_odom_test "
			   "config/kitti_odom/kitti_config00-02.yaml "
			   "/path/to/kitti/odometry/sequences/00/ \n");
		return 1;
	}

	string config_file = argv[1];
	printf("config_file: %s\n", argv[1]);
	string sequence = argv[2];
	printf("read sequence: %s\n", argv[2]);
	string dataPath = sequence + "/";

	readParameters(config_file);
	estimator.setParameter();
	registerPub(n);

	const char *export_dir_value = std::getenv("KITTI_EXPORT_TRACK_DIR");
	const std::string export_dir = export_dir_value == nullptr
		? std::string()
		: std::string(export_dir_value);
	const std::set<size_t> export_frames = parseExportFrames(
		std::getenv("KITTI_EXPORT_TRACK_FRAMES"));
	const bool export_track_images = !export_dir.empty() && !export_frames.empty();
	if (export_track_images)
	{
		if (!cv::utils::fs::createDirectories(export_dir))
		{
			std::cerr << "Cannot create track image export directory: "
			          << export_dir << std::endl;
			return 1;
		}
		std::cout << "Export tracking images to " << export_dir
		          << " for " << export_frames.size() << " selected frames"
		          << std::endl;
	}

	// load image list
	FILE* file;
	file = std::fopen((dataPath + "times.txt").c_str() , "r");
	if(file == NULL){
	    printf("cannot find file: %stimes.txt\n", dataPath.c_str());
	    // ROS_BREAK();
	    return 0;          
	}
	double imageTime;
	vector<double> imageTimeList;
	while ( fscanf(file, "%lf", &imageTime) != EOF)
	{
	    imageTimeList.push_back(imageTime);
	}
	std::fclose(file);

	string leftImagePath, rightImagePath;
	string leftImageDir = dataPath + "image_0/";
	string rightImageDir = dataPath + "image_1/";
	if (!cv::utils::fs::exists(leftImageDir) ||
		!cv::utils::fs::exists(rightImageDir))
	{
		leftImageDir = dataPath + "image_2/";
		rightImageDir = dataPath + "image_3/";
	}
	cv::Mat imLeft, imRight, imLeftColor;
	FILE* outFile;
	outFile = fopen((OUTPUT_FOLDER + "/vio.txt").c_str(),"w");
	if(outFile == NULL)
		printf("Output path dosen't exist: %s\n", OUTPUT_FOLDER.c_str());

	for (size_t i = 0; i < imageTimeList.size(); i++)
	{	
			printf("\nprocess image %d\n", (int)i);
			stringstream ss;
			ss << setfill('0') << setw(6) << i;
			leftImagePath = leftImageDir + ss.str() + ".png";
			rightImagePath = rightImageDir + ss.str() + ".png";
			//printf("%lu  %f \n", i, imageTimeList[i]);
			//printf("%s\n", leftImagePath.c_str() );
			//printf("%s\n", rightImagePath.c_str() );

			imLeft = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE );
			imLeftColor = cv::imread(leftImagePath, cv::IMREAD_COLOR);
			// sensor_msgs::msg::ImagePtr 
			auto imLeftMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", imLeft).toImageMsg();
			imLeftMsg->header.stamp = rclcpp::Time(imageTimeList[i]);
			pubLeftImage->publish(*imLeftMsg);

			imRight = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE );
			// sensor_msgs::msg::ImagePtr 
			auto imRightMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", imRight).toImageMsg();
			imRightMsg->header.stamp = rclcpp::Time(imageTimeList[i]);
			pubRightImage->publish(*imRightMsg);


			estimator.inputImage(imageTimeList[i], imLeft, imRight, imLeftColor);
			if (export_track_images && export_frames.count(i) != 0)
			{
				const cv::Mat track_image = FEATURE_TRACKER_TYPE == 3
					? estimator.yoloPointLightGlueFeatureTracker.getTrackImage()
					: estimator.featureTracker.getTrackImage();
				stringstream export_name;
				export_name << export_dir << "/frame_" << setfill('0')
				            << setw(6) << i << "_dynamic_filter.png";
				if (track_image.empty() || !cv::imwrite(export_name.str(), track_image))
					std::cerr << "Failed to export tracking image: "
					          << export_name.str() << std::endl;
				else
					std::cout << "Exported tracking image: "
					          << export_name.str() << std::endl;
			}
			
			Eigen::Matrix<double, 4, 4> pose;
			estimator.getPoseInWorldFrame(pose);
			Eigen::Vector3d corrected_p;
			Eigen::Quaterniond corrected_q;
			independentLoopClosure().recordOdometry(
				imageTimeList[i], pose.block<3, 1>(0, 3),
				Eigen::Quaterniond(pose.block<3, 3>(0, 0)),
				Eigen::Vector3d::Zero());
			independentLoopClosure().correctPose(
				pose.block<3, 1>(0, 3),
				Eigen::Quaterniond(pose.block<3, 3>(0, 0)),
				corrected_p, corrected_q);
			pose.block<3, 3>(0, 0) = corrected_q.toRotationMatrix();
			pose.block<3, 1>(0, 3) = corrected_p;
			if(outFile != NULL)
				fprintf (outFile, "%f %f %f %f %f %f %f %f %f %f %f %f\n",pose(0,0), pose(0,1), pose(0,2),pose(0,3),
																	       pose(1,0), pose(1,1), pose(1,2),pose(1,3),
																	       pose(2,0), pose(2,1), pose(2,2),pose(2,3));
			
			//cv::imshow("leftImage", imLeft);
			//cv::imshow("rightImage", imRight);
			//cv::waitKey(2);
			if (export_track_images && i >= *export_frames.rbegin())
				break;
	}
	if(outFile != NULL)
		fclose (outFile);
	estimator.shutdown();
	if (LOOP_CLOSURE_ENABLE)
	{
		independentLoopClosure().waitUntilIdle();
		independentLoopClosure().saveTrajectory(OUTPUT_FOLDER + "/loop_pose_graph.tum");
		independentLoopClosure().saveOptimizedTrajectory(OUTPUT_FOLDER + "/loop_optimized.csv");
	}
	shutdownVisualization();
	pubRightImage.reset();
	pubLeftImage.reset();
	n.reset();
	rclcpp::shutdown();
	return 0;
}

/*******************************************************
 * Weak adjacent-frame z-motion prior for stereo-only KITTI odometry.
 *******************************************************/

#pragma once

#include <ceres/ceres.h>

struct KittiZMotionFactor
{
    explicit KittiZMotionFactor(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T *const pose_i, const T *const pose_j, T *residual) const
    {
        residual[0] = T(weight_) * (pose_j[2] - pose_i[2]);
        return true;
    }

    static ceres::CostFunction *Create(double weight)
    {
        return new ceres::AutoDiffCostFunction<KittiZMotionFactor, 1, 7, 7>(
            new KittiZMotionFactor(weight));
    }

    double weight_;
};

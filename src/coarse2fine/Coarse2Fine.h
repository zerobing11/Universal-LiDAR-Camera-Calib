#ifndef COARSE2FINE_H
#define COARSE2FINE_H

#include <vector>
#include <array>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class Coarse2Fine {
public:
    Coarse2Fine();
    ~Coarse2Fine();

    // 执行单帧优化
    bool add(const std::vector<Eigen::Vector4f>& camera_planes,
             const Eigen::Matrix3d& Coarse_Rcl,
             const Eigen::Vector3d& Coarse_tcl,
             const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& filtered_clouds,
             Eigen::Matrix3d& Optimized_Rcl,
             Eigen::Vector3d& Optimized_tcl);

    // 使用所有添加的帧执行联合优化
    bool JointOptimize(Eigen::Matrix3d& Final_Rcl, Eigen::Vector3d& Final_tcl);

    // 将相机系平面在单帧外参下转换到雷达系
    std::array<Eigen::Vector4f, 3> TransformPlanesToLidarSingleFrame(
        const std::vector<Eigen::Vector4d>& planes_cam,
        const Eigen::Matrix3d& Rcl_opt,
        const Eigen::Vector3d& tcl_opt);

    const std::array<Eigen::Vector4f, 3>& GetLidarPlanes() const;

private:

    //用于最后的联合优化
    struct FrameData {
        std::vector<Eigen::Vector4d> camera_planes;
        std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> clouds;
        Eigen::Matrix3d initial_Rcl;
        Eigen::Vector3d initial_tcl;
    };
    std::vector<FrameData> frames_data_;

    //相机系下平面方程转到雷达系下
    std::array<Eigen::Vector4f, 3> lidar_planes_;

    // 单帧优化实现
    bool OptimizeSingleFrame(const std::vector<Eigen::Vector4d>& planes_cam,
                             const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& clouds,
                             const Eigen::Matrix3d& Rcl_init,
                             const Eigen::Vector3d& tcl_init,
                             Eigen::Matrix3d& Rcl_opt,
                             Eigen::Vector3d& tcl_opt);
};

#endif // COARSE2FINE_H

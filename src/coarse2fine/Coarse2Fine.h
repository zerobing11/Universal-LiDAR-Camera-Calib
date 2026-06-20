#ifndef COARSE2FINE_H
#define COARSE2FINE_H

#include <vector>
#include <array>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class Coarse2Fine {
public:
    Coarse2Fine();
    ~Coarse2Fine();

    bool add(const std::vector<Eigen::Vector4f>& camera_planes,
             const Eigen::Matrix3d& Coarse_Rcl,
             const Eigen::Vector3d& Coarse_tcl,
             const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& filtered_clouds,
             Eigen::Matrix3d& Optimized_Rcl,
             Eigen::Vector3d& Optimized_tcl);

    bool JointOptimize(Eigen::Matrix3d& Final_Rcl, Eigen::Vector3d& Final_tcl);

    std::array<Eigen::Vector4f, 3> TransformPlanesToLidarSingleFrame(
        const std::vector<Eigen::Vector4d>& planes_cam,
        const Eigen::Matrix3d& Rcl_opt,
        const Eigen::Vector3d& tcl_opt);

    const std::array<Eigen::Vector4f, 3>& GetLidarPlanes() const;

private:

    struct FrameData {
        std::vector<Eigen::Vector4d> camera_planes;
        std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> clouds;
        Eigen::Matrix3d initial_Rcl;
        Eigen::Vector3d initial_tcl;
    };
    std::vector<FrameData> frames_data_;

    std::array<Eigen::Vector4f, 3> lidar_planes_;

    bool OptimizeSingleFrame(const std::vector<Eigen::Vector4d>& planes_cam,
                             const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& clouds,
                             const Eigen::Matrix3d& Rcl_init,
                             const Eigen::Vector3d& tcl_init,
                             Eigen::Matrix3d& Rcl_opt,
                             Eigen::Vector3d& tcl_opt);
};

#endif // COARSE2FINE_H

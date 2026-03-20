#ifndef _LIDAR_SORT_H_
#define _LIDAR_SORT_H_

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/common/centroid.h>
#include <pcl/kdtree/kdtree_flann.h>

class LidarSort {
private:

    struct PlaneInfo {
        Eigen::Vector4f coeffs;
        Eigen::Vector4f centroid;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    };

    //输入参数
        //一些区域增长参数
    double leaf_size_;
    int min_points_per_plane_;
    int normal_k_search_;
    double smoothness_threshold_deg_;
    double curvature_threshold_;
    double ground_smoothness_threshold_deg_;
    double ground_curvature_threshold_;
    int rg_neighbor_k_;
    bool extract_ground_;
         //三维标定板验证参数
    double centroid_distance_target_;
    double plane_orthogonality_threshold_;
          //原始场景点云提取参数
    double extraction_radius_;
    float dist_threshold_;

    //最后的输出
    std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> final_clouds_;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> all_candidate_planes_;

    void ResetFinalClouds();
    //区域增长算法提取地面平面
    std::vector<pcl::PointIndices> performRegionGrowing(
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
        int min_cluster_size,
        int neighbor_k,
        double smoothness_deg,
        double curvature_thresh) const;

    //提取地面平面
    pcl::PointCloud<pcl::PointXYZI>::Ptr extractGroundPlane(
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
        int min_points,
        int neighbor_k,
        double smoothness_deg,
        double curvature_thresh,
        pcl::PointCloud<pcl::PointXYZI>::Ptr &non_ground_out) const;

    //提取屋顶平面
    pcl::PointCloud<pcl::PointXYZI>::Ptr extractRoofPlane(
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
        int min_points,
        int neighbor_k,
        double smoothness_deg,
        double curvature_thresh,
        pcl::PointCloud<pcl::PointXYZI>::Ptr &non_roof_out) const;

    //选择最佳三个平面
    std::vector<PlaneInfo> selectBestThreePlanes(
        const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &all_planes,
        double target_dist,
        double orthogonality_threshold) const;

    //提取最终点云
    std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> extractFinalPoints(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr &raw_cloud,
        const std::vector<PlaneInfo> &selected_planes,
        float dist_threshold,
        double radius) const;

    //排序平面
    std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> sort_planes(
        const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> &final_clouds) const;

public:
    LidarSort(double leaf_size,
              int min_points_per_plane,
              int normal_k_search,
              double smoothness_threshold_deg,
              double curvature_threshold,
              double centroid_distance_target,
              double extraction_radius,
              double plane_orthogonality_threshold,
              int rg_neighbor_k,
              double ground_smoothness_threshold_deg,
              double ground_curvature_threshold,
              bool extract_ground,
              float dist_threshold);

    bool add(const std::string &lidar_name);

    void GetFinalClouds(std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> &final_clouds) const;

    void GetCandidatePlanes(std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &candidate_planes) const;
};

#endif


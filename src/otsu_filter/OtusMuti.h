#ifndef _OTSU_MUTI_H_
#define _OTSU_MUTI_H_

#include <vector>
#include <array>
#include <limits>
#include <cmath>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class OtusMuti {
private:
    //过滤后的点云
    std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> filtered_clouds_;

    void ClearFilteredClouds();
    //计算大津法阈值    
    static bool ComputeOtsuThreshold(
        const pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr &cloud,
        double &intensity_threshold);
    //多级大津法
    static bool ComputeMultiLevelOtsu(
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr &working_cloud,
        double &intensity_threshold,
        const double min_ratio,
        size_t &iterations,
        size_t &removed_points);
    void FilterClouds(
        const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> &input_clouds,
        const double min_ratio);

public:
    OtusMuti();

    void add(
        const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> &input_clouds,
        double min_ratio = 0.2);

    void GetFilteredClouds(
        std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> &output_clouds) const;
};

#endif


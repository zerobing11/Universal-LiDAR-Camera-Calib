#ifndef _PCLlib_
#define _PCLlib_

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#include <time.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <numeric>
#include <string>
#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <array>
#include <limits>
#include <atomic>
#include <thread>
#include <dirent.h>
#include <sys/stat.h>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/print.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"

#include <Eigen/Core>
#include <opencv2/core/types.hpp>


using namespace std;

struct LineEquation {
    Eigen::Vector3d point;
    Eigen::Vector3d direction;
};

//基准坐标系
struct PlaneBasis {
    Eigen::Vector3f origin;
    Eigen::Vector3f u;
    Eigen::Vector3f v;
    Eigen::Vector4f plane;
};

pcl::PointCloud<pcl::PointXYZI> Load_ply(const std::string &ply_file) {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    if (pcl::io::loadPLYFile<pcl::PointXYZI>(ply_file, cloud) != 0) {
        std::cerr << "Failed to load PLY file: " << ply_file << std::endl;
    }
    return cloud;
}

bool SavePLY(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
             const std::string &ply_path,
             bool binary = true) {
    if (!cloud) {
        std::cerr << "SavePLY failed: cloud is null, path=" << ply_path << std::endl;
        return false;
    }
    int ret = binary
                  ? pcl::io::savePLYFileBinary(ply_path, *cloud)
                  : pcl::io::savePLYFile(ply_path, *cloud);
    if (ret != 0) {
        std::cerr << "SavePLY failed: ret=" << ret << ", path=" << ply_path << std::endl;
        return false;
    }
    return true;
}

bool SavePLY(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &cloud,
             const std::string &ply_path,
             bool binary = true) {
    if (!cloud) {
        std::cerr << "SavePLY failed: cloud is null, path=" << ply_path << std::endl;
        return false;
    }
    int ret = binary
                  ? pcl::io::savePLYFileBinary(ply_path, *cloud)
                  : pcl::io::savePLYFile(ply_path, *cloud);
    if (ret != 0) {
        std::cerr << "SavePLY failed: ret=" << ret << ", path=" << ply_path << std::endl;
        return false;
    }
    return true;
}

bool ensureDirectory(const std::string &path) {
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }
    if (mkdir(path.c_str(), 0755) != 0) {
        perror(("Failed to create directory: " + path).c_str());
        return false;
    }
    return true;
}

bool isPlyFile(const std::string &file) {
    if (file.length() < 4)
        return false;
    std::string suffix = file.substr(file.length() - 4);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
    return suffix == ".ply";
}

std::vector<std::string> collectPlyFiles(const std::string &directory) {
    std::vector<std::string> files;
    DIR *dir = opendir(directory.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open directory " << directory << std::endl;
        return files;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (isPlyFile(filename)) {
            files.push_back(filename.substr(0, filename.length() - 4));
        }
    }
    closedir(dir);
    std::sort(files.begin(), files.end());
    return files;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr ExtractSpherePoints(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud,
    const pcl::PointXYZ& seed,
    double extraction_radius)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr out(new pcl::PointCloud<pcl::PointXYZI>());
    if (!cloud || cloud->empty() || extraction_radius <= 0.0) {
        out->width = 0;
        out->height = 1;
        out->is_dense = true;
        return out;
    }
    const double r2 = extraction_radius * extraction_radius;
    out->points.reserve(cloud->size());
    for (const auto& pt : cloud->points) {
        const double dx = pt.x - seed.x;
        const double dy = pt.y - seed.y;
        const double dz = pt.z - seed.z;
        if (dx * dx + dy * dy + dz * dz <= r2) {
            out->points.push_back(pt);
        }
    }
    out->width = static_cast<uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = true;
    return out;
}

void CheckBoardPlane(std::array<Eigen::Vector4f, 3> &p) {
    if (p.size() == 3) {
        Eigen::Vector4f p1 = p[0];
        Eigen::Vector4f p2 = p[1];
        Eigen::Vector4f p3 = p[2];

        float dot12 = p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2];
        float dot13 = p1[0] * p3[0] + p1[1] * p3[1] + p1[2] * p3[2];
        float dot23 = p2[0] * p3[0] + p2[1] * p3[1] + p2[2] * p3[2];
        float angle12 = acos(abs(dot12)) * 180.0 / M_PI;
        float angle13 = acos(abs(dot13)) * 180.0 / M_PI;
        float angle23 = acos(abs(dot23)) * 180.0 / M_PI;
        // std::cout << "angle_plane12: " << angle12 << "°" << std::endl;
        // std::cout << "angle_plane13: " << angle13 << "°" << std::endl;
        // std::cout << "angle_plane23: " << angle23 << "°" << std::endl;
    }
}

void publishPointCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud,
                       const ros::Publisher &publisher) {
    if (!ros::ok() || !cloud)
        return;
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.frame_id = "/velodyne";
    msg.header.stamp = ros::Time::now();
    publisher.publish(msg);
}

void publishPointCloud(const pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr &cloud,
                       const ros::Publisher &publisher) {
    if (!ros::ok() || !cloud)
        return;
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.frame_id = "/velodyne";
    msg.header.stamp = ros::Time::now();
    publisher.publish(msg);
}

void transformCam3dToLidar3d(const std::vector<cv::Point3f>& cam_points,
                             const Eigen::Matrix3d& Rcl,
                             const Eigen::Vector3d& tcl,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& out_cloud) {
    if (!out_cloud) {
        out_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    }
    out_cloud->clear();
    if (cam_points.empty()) {
        out_cloud->width = 0;
        out_cloud->height = 1;
        out_cloud->is_dense = true;
        return;
    }
    out_cloud->points.reserve(cam_points.size());
    for (const auto& p_cam : cam_points) {
        Eigen::Vector3d p_c(p_cam.x, p_cam.y, p_cam.z);
        // P_cam = Rcl * P_lidar + tcl  =>  P_lidar = Rcl^T * (P_cam - tcl)
        Eigen::Vector3d p_l = Rcl.transpose() * (p_c - tcl);
        pcl::PointXYZI p;
        p.x = static_cast<float>(p_l.x());
        p.y = static_cast<float>(p_l.y());
        p.z = static_cast<float>(p_l.z());
        p.intensity = 0.0f;
        out_cloud->points.push_back(p);
    }
    out_cloud->width = out_cloud->points.size();
    out_cloud->height = 1;
    out_cloud->is_dense = true;
}


void waitForEnter(const std::string &prompt, bool enable_wait) {
    if (!enable_wait)
        return;
    std::cout << prompt;
    std::string dummy;
    std::getline(std::cin, dummy);
}

void publishUntilEnter(const std::function<void()> &publish_fn,
                       const std::string &prompt,
                       bool enable_wait,
                       double rate_hz = 10.0) {
    if (!enable_wait) {
        publish_fn();
        return;
    }
    std::cout << prompt;
    std::atomic<bool> stop(false);
    std::thread th([&]() {
        ros::Rate r(rate_hz);
        while (ros::ok() && !stop.load()) {
            publish_fn();
            ros::spinOnce();
            r.sleep();
        }
    });
    std::string dummy;
    std::getline(std::cin, dummy);
    stop.store(true);
    if (th.joinable())
        th.join();
}

bool classifyAndPublish(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                        const std::vector<Eigen::Vector4f>& planes,
                        const ros::Publisher& publisher,
                        bool wait_enter,
                        const std::string& prompt)
{
    if (!cloud || planes.empty())
        return false;
    pcl::PointCloud<pcl::PointXYZI>::Ptr classified(new pcl::PointCloud<pcl::PointXYZI>(*cloud));
    for (auto& pt : classified->points)
    {
        double best_dist = std::numeric_limits<double>::max();
        int best_idx = -1;
        for (size_t pid = 0; pid < planes.size(); ++pid)
        {
            const auto& p = planes[pid];
            double denom = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
            if (denom < 1e-9)
                continue;
            double dist = std::abs(p[0]*pt.x + p[1]*pt.y + p[2]*pt.z + p[3]) / denom;
            if (dist < best_dist)
            {
                best_dist = dist;
                best_idx = static_cast<int>(pid);
            }
        }
        if (best_idx >= 0)
            pt.intensity = static_cast<float>(best_idx * 40); // 用不同强度区分平面
    }
    publishPointCloud(classified, publisher);
    waitForEnter(prompt, wait_enter);
    return true;
}

void projectPointToPlane(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud,
                         const Eigen::Vector4f& plane,
                         pcl::PointCloud<pcl::PointXYZINormal>::Ptr& projected_cloud)
{
    if (!projected_cloud)
        projected_cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    projected_cloud->clear();
    if (!cloud)
        return;
    projected_cloud->reserve(cloud->size());
    for (const auto& pt : cloud->points)
    {
        const double denom = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z;
        if (std::abs(denom) < 1e-10)
        {
            continue; 
        }
        const double t = -static_cast<double>(plane[3]) / denom; 
        pcl::PointXYZINormal projected;
        projected.x = static_cast<float>(t * pt.x);
        projected.y = static_cast<float>(t * pt.y);
        projected.z = static_cast<float>(t * pt.z);
        projected.intensity = pt.intensity;
        projected.curvature = pt.curvature;
        projected.normal_x = pt.normal_x;
        projected.normal_y = pt.normal_y;
        projected.normal_z = pt.normal_z;
        projected_cloud->push_back(projected);
    }
    projected_cloud->width = projected_cloud->size();
    projected_cloud->height = 1;
}


std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> FilterCloudByLines(
    const std::vector<std::pair<LineEquation, LineEquation>>& line_equations,
    const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& plane_clouds,
    double radius,
    int num_remove_far = 0)
{
    std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> filtered_clouds;

    for (size_t i = 0; i < 3; ++i) {
        filtered_clouds[i].reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        if (!plane_clouds[i] || i >= line_equations.size()) {
            continue;
        }

        const auto& l1 = line_equations[i].first;
        const auto& l2 = line_equations[i].second;
        
        Eigen::Vector3d dir1 = l1.direction.normalized();
        Eigen::Vector3d dir2 = l2.direction.normalized();
        Eigen::Vector3d p1 = l1.point;
        Eigen::Vector3d p2 = l2.point;

        std::vector<pcl::PointXYZINormal> kept_points;
        std::vector<double> min_dists;
        kept_points.reserve(plane_clouds[i]->size());
        min_dists.reserve(plane_clouds[i]->size());

        for (const auto& pt : plane_clouds[i]->points) {
            Eigen::Vector3d p(pt.x, pt.y, pt.z);

            double d1 = ((p - p1).cross(dir1)).norm();
            double d2 = ((p - p2).cross(dir2)).norm();

            // 移除交线半径内的点（圆柱体剔除）
            if (d1 > radius && d2 > radius) {
                kept_points.push_back(pt);
                min_dists.push_back(std::min(d1, d2));
            }
        }
        
        if (kept_points.empty()) {
            filtered_clouds[i]->width = 0;
            filtered_clouds[i]->height = 1;
            filtered_clouds[i]->is_dense = true;
            continue;
        }

        // 移除最远的num个点
        std::vector<size_t> indices(kept_points.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b){
            return min_dists[a] < min_dists[b];
        });

        size_t final_count = kept_points.size();
        if (num_remove_far > 0) {
            if (static_cast<size_t>(num_remove_far) >= kept_points.size()) {
                final_count = 0;
            } else {
                final_count = kept_points.size() - num_remove_far;
            }
        }
        
        filtered_clouds[i]->points.reserve(final_count);
        for(size_t k=0; k<final_count; ++k) {
            filtered_clouds[i]->points.push_back(kept_points[indices[k]]);
        }

        filtered_clouds[i]->width = filtered_clouds[i]->points.size();
        filtered_clouds[i]->height = 1;
        filtered_clouds[i]->is_dense = true;
    }

    return filtered_clouds;
}

void voxelDownsample(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                     float leaf_size,
                     pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud)
{
    if (!output_cloud)
        output_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    output_cloud->clear();
    if (!input_cloud || input_cloud->empty())
        return;

    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel.setInputCloud(input_cloud);
    voxel.filter(*output_cloud);
    output_cloud->width = output_cloud->size();
    output_cloud->height = 1;
}

void voxelDownsample(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& input_cloud,
                     float leaf_size,
                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr& output_cloud)
{
    if (!output_cloud)
        output_cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    output_cloud->clear();
    if (!input_cloud || input_cloud->empty())
        return;

    pcl::VoxelGrid<pcl::PointXYZINormal> voxel;
    voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel.setInputCloud(input_cloud);
    voxel.filter(*output_cloud);
    output_cloud->width = output_cloud->size();
    output_cloud->height = 1;
}

void voxelDownsample(const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& input_clouds,
                     float leaf_size,
                     std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& output_clouds)
{
    for (size_t i = 0; i < output_clouds.size(); ++i) {
        voxelDownsample(input_clouds[i], leaf_size, output_clouds[i]);
    }
}

void setPlaneIntensity(std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& plane_clouds,
                       const std::array<float, 3>& intensities = {0.0f, 40.0f, 80.0f})
{
    for (size_t i = 0; i < plane_clouds.size(); ++i) {
        auto& cloud = plane_clouds[i];
        if (!cloud || cloud->empty())
            continue;
        for (auto& pt : cloud->points) {
            pt.intensity = intensities[i];
        }
    }
}


bool ComputePlaneFromPoints(const Eigen::Vector3d& p1,
                            const Eigen::Vector3d& p2,
                            const Eigen::Vector3d& p3,
                            Eigen::Vector4d& plane_out)
{
    Eigen::Vector3d v1 = p2 - p1;
    Eigen::Vector3d v2 = p3 - p1;
    Eigen::Vector3d n = v1.cross(v2);
    double n_norm = n.norm();
    if(n_norm < 1e-9)
    {
        return false;
    }
    Eigen::Vector3d n_unit = n / n_norm;
    double d = -n_unit.dot(p1);
    plane_out.head<3>() = n_unit;
    plane_out[3] = d;
    return true;
}

double SignedDistanceToPlane(const Eigen::Vector4d& plane,
                             const Eigen::Vector3d& p)
{
    return plane.head<3>().dot(p) + plane[3];
}

struct MlesacResult
{
    Eigen::Vector4f plane;
    double score = 0.0;
    bool success = false;
};
MlesacResult RunPlaneMlesac(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud,
                            double sigma = 0.02,
                            double gamma = 0.95,
                            int max_iterations = 200)
{
    // 内点高斯分布
    MlesacResult result;
    if (!cloud || cloud->size() < 3) {
        return result;
    }
    const double eps = 1e-12;
    const double sigma_sq = sigma * sigma;
    const double inv_gauss_norm = 1.0 / (std::sqrt(2.0 * M_PI) * sigma); // 高斯系数
    // 外点均匀分布
    double diag_len =2.0 * sigma;
    const double p_out = 1.0 / diag_len;

    std::mt19937 gen(static_cast<unsigned>(std::random_device{}()));
    std::uniform_int_distribution<int> dist_idx(0, static_cast<int>(cloud->size()) - 1);

    double best_loglik = -std::numeric_limits<double>::infinity();
    Eigen::Vector4d best_plane;

    auto sample_unique = [&](int n)->std::array<int,3>
    {
        std::array<int,3> idx;
        do { idx[0] = dist_idx(gen); idx[1] = dist_idx(gen); idx[2] = dist_idx(gen); }
        while(!(idx[0]!=idx[1] && idx[0]!=idx[2] && idx[1]!=idx[2]));
        return idx;
    };

    for(int iter=0; iter<max_iterations; ++iter)
    {
        auto idx = sample_unique(3);
        Eigen::Vector4d plane;
        Eigen::Vector3d p1(cloud->points[idx[0]].x, cloud->points[idx[0]].y, cloud->points[idx[0]].z);
        Eigen::Vector3d p2(cloud->points[idx[1]].x, cloud->points[idx[1]].y, cloud->points[idx[1]].z);
        Eigen::Vector3d p3(cloud->points[idx[2]].x, cloud->points[idx[2]].y, cloud->points[idx[2]].z);
        if(!ComputePlaneFromPoints(p1, p2, p3, plane))
        {
            continue;
        }

        double loglik = 0.0;
        for(const auto& pt : cloud->points)
        {
            Eigen::Vector3d p(pt.x, pt.y, pt.z);
            double d = std::abs(SignedDistanceToPlane(plane, p));
            // 混合分布似然：gamma*Gaussian + (1-gamma)*Uniform
            double pin = inv_gauss_norm * std::exp(-d*d/(2.0*sigma_sq));
            double mix = gamma * pin + (1.0 - gamma) * p_out + eps;
            loglik += std::log(mix);
        }

        if(loglik > best_loglik)
        {
            best_loglik = loglik;
            best_plane = plane;
        }
    }

    if(!std::isfinite(best_loglik))
    {
        return result;
    }

    if (best_plane[3] < 0.0) {
        best_plane = -best_plane;
    }

    double accum = 0.0;
    for(const auto& pt : cloud->points)
    {
        Eigen::Vector3d p(pt.x, pt.y, pt.z);
        double d = std::abs(SignedDistanceToPlane(best_plane, p));
        accum += std::exp(-d*d/(2.0*sigma_sq));
    }
    double score = 100.0 * accum / static_cast<double>(cloud->size());

    result.plane = best_plane.cast<float>();
    result.score = score;
    result.success = true;

    auto& pts = cloud->points;
    pts.erase(
        std::remove_if(pts.begin(), pts.end(),
                       [&](const pcl::PointXYZINormal& pt) {
                           Eigen::Vector3d p(pt.x, pt.y, pt.z);
                           double d = std::abs(SignedDistanceToPlane(best_plane, p));
                           return d > sigma;
                       }),
        pts.end());
    cloud->width = static_cast<uint32_t>(pts.size());
    cloud->height = 1;

    return result;
}

#endif

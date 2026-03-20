#include "LidarSort.h"

using namespace std;

void LidarSort::ResetFinalClouds() {
    for (auto &cloud: final_clouds_) {
        cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    }
}
//区域生长提取平面
std::vector<pcl::PointIndices> LidarSort::performRegionGrowing(
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
    int min_cluster_size,
    int neighbor_k,
    double smoothness_deg,
    double curvature_thresh) const {
    if (!cloud || cloud->empty())
        return {};
    pcl::search::Search<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(std::max(1, normal_k_search_));
    normal_estimator.compute(*normals);

    pcl::RegionGrowing<pcl::PointXYZI, pcl::Normal> reg;
    reg.setMinClusterSize(min_cluster_size);
    reg.setMaxClusterSize(1000000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(std::max(1, neighbor_k));
    reg.setInputCloud(cloud);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(smoothness_deg / 180.0 * M_PI);
    reg.setCurvatureThreshold(curvature_thresh);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);
    return clusters;
}
//区域生长剔除地面
pcl::PointCloud<pcl::PointXYZI>::Ptr LidarSort::extractGroundPlane(
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
    int min_points,
    int neighbor_k,
    double smoothness_deg,
    double curvature_thresh,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &non_ground_out) const {
    pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    if (!cloud || cloud->empty()) {
        non_ground_out.reset(new pcl::PointCloud<pcl::PointXYZI>());
        return ground_cloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr sorted_cloud(new pcl::PointCloud<pcl::PointXYZI>(*cloud));
    std::sort(sorted_cloud->points.begin(), sorted_cloud->points.end(),
              [](const pcl::PointXYZI &a, const pcl::PointXYZI &b) { return a.z < b.z; });

    int num_lpr = std::max(1, static_cast<int>(sorted_cloud->size() * 0.01));
    double avg_lpr_z = 0.0;
    for (int k = 0; k < num_lpr; ++k)
        avg_lpr_z += sorted_cloud->points[k].z;
    avg_lpr_z /= num_lpr;

    double seed_height_threshold = avg_lpr_z + 0.3;
    std::vector<int> ground_seeds;
    for (size_t k = 0; k < cloud->size(); ++k) {
        if (cloud->points[k].z < seed_height_threshold)
            ground_seeds.push_back(static_cast<int>(k));
    }

    std::vector<pcl::PointIndices> clusters = performRegionGrowing(
        cloud, min_points, neighbor_k, smoothness_deg, curvature_thresh);

    int best_ground_idx = -1;
    size_t max_overlap = 0;

    std::vector<bool> is_seed(cloud->size(), false);
    for (int idx: ground_seeds) is_seed[idx] = true;

    for (size_t i = 0; i < clusters.size(); ++i) {
        size_t overlap = 0;
        for (int idx: clusters[i].indices) {
            if (is_seed[idx]) overlap++;
        }
        if (overlap > max_overlap) {
            max_overlap = overlap;
            best_ground_idx = static_cast<int>(i);
        }
    }

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(cloud);
    extract.setNegative(false);

    if (best_ground_idx != -1) {
        pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices(clusters[best_ground_idx]));
        extract.setIndices(ground_inliers);
        extract.filter(*ground_cloud);

        extract.setNegative(true);
        extract.filter(*non_ground_out);
    } else {
        non_ground_out.reset(new pcl::PointCloud<pcl::PointXYZI>(*cloud));
    }
    return ground_cloud;
}
//区域生长剔除天花板
pcl::PointCloud<pcl::PointXYZI>::Ptr LidarSort::extractRoofPlane(
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
    int min_points,
    int neighbor_k,
    double smoothness_deg,
    double curvature_thresh,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &non_roof_out) const {
    pcl::PointCloud<pcl::PointXYZI>::Ptr roof_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    if (!cloud || cloud->empty()) {
        non_roof_out.reset(new pcl::PointCloud<pcl::PointXYZI>());
        return roof_cloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr sorted_cloud(new pcl::PointCloud<pcl::PointXYZI>(*cloud));
    std::sort(sorted_cloud->points.begin(), sorted_cloud->points.end(),
              [](const pcl::PointXYZI &a, const pcl::PointXYZI &b) { return a.z > b.z; });

    int num_lpr = std::max(1, static_cast<int>(sorted_cloud->size() * 0.01));
    double avg_lpr_z = 0.0;
    for (int k = 0; k < num_lpr; ++k)
        avg_lpr_z += sorted_cloud->points[k].z;
    avg_lpr_z /= num_lpr;

    double seed_height_threshold = avg_lpr_z - 0.3;
    std::vector<int> roof_seeds;
    for (size_t k = 0; k < cloud->size(); ++k) {
        if (cloud->points[k].z > seed_height_threshold)
            roof_seeds.push_back(static_cast<int>(k));
    }

    std::vector<pcl::PointIndices> clusters = performRegionGrowing(
        cloud, min_points, neighbor_k, smoothness_deg, curvature_thresh);

    int best_roof_idx = -1;
    size_t max_overlap = 0;

    std::vector<bool> is_seed(cloud->size(), false);
    for (int idx: roof_seeds) is_seed[idx] = true;

    for (size_t i = 0; i < clusters.size(); ++i) {
        size_t overlap = 0;
        for (int idx: clusters[i].indices) {
            if (is_seed[idx]) overlap++;
        }
        if (overlap > max_overlap) {
            max_overlap = overlap;
            best_roof_idx = static_cast<int>(i);
        }
    }

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(cloud);
    extract.setNegative(false);

    if (best_roof_idx != -1) {
        pcl::PointIndices::Ptr roof_inliers(new pcl::PointIndices(clusters[best_roof_idx]));
        extract.setIndices(roof_inliers);
        extract.filter(*roof_cloud);

        extract.setNegative(true);
        extract.filter(*non_roof_out);
    } else {
        non_roof_out.reset(new pcl::PointCloud<pcl::PointXYZI>(*cloud));
    }
    return roof_cloud;
}
//挑选出三维标定板平面
std::vector<LidarSort::PlaneInfo> LidarSort::selectBestThreePlanes(
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &all_planes,
    double target_dist,
    double orthogonality_threshold) const {
    if (all_planes.size() < 3) return {};

    std::vector<PlaneInfo> planes_info;
    planes_info.reserve(all_planes.size());
    for (const auto &cloud: all_planes) {
        if (!cloud || cloud->empty())
            continue;
        PlaneInfo info;
        info.cloud = cloud;

        pcl::compute3DCentroid(*cloud, info.centroid);

        Eigen::Vector4f plane_parameters;
        float curvature;
        pcl::computePointNormal(*cloud, plane_parameters, curvature);
        info.coeffs = plane_parameters;
        planes_info.push_back(info);
    }

    if (planes_info.size() < 3)
        return {};

    struct Triplet {
        int i, j, k;
        double E;
    };
    std::vector<Triplet> valid_triplets;

    for (size_t i = 0; i < planes_info.size(); ++i) {
        for (size_t j = i + 1; j < planes_info.size(); ++j) {
            for (size_t k = j + 1; k < planes_info.size(); ++k) {
                Eigen::Vector3f n1 = planes_info[i].coeffs.head<3>();
                Eigen::Vector3f n2 = planes_info[j].coeffs.head<3>();
                Eigen::Vector3f n3 = planes_info[k].coeffs.head<3>();
                //正交阈值要满足
                auto calc_angle_deg = [](const Eigen::Vector3f &a, const Eigen::Vector3f &b) {
                    Eigen::Vector3f na = a;
                    Eigen::Vector3f nb = b;
                    const double na_norm = na.norm();
                    const double nb_norm = nb.norm();
                    if (na_norm > 1e-9) na /= static_cast<float>(na_norm);
                    if (nb_norm > 1e-9) nb /= static_cast<float>(nb_norm);
                    double dot = na.dot(nb);
                    if (dot > 1.0) dot = 1.0;
                    if (dot < -1.0) dot = -1.0;
                    return std::acos(dot) * 180.0 / M_PI;
                };

                double angle12 = calc_angle_deg(n1, n2);
                double angle13 = calc_angle_deg(n1, n3);
                double angle23 = calc_angle_deg(n2, n3);

                double term1 = std::abs(90.0 - angle12);
                double term2 = std::abs(90.0 - angle13);
                double term3 = std::abs(90.0 - angle23);

                double E = (term1 + term2 + term3) / 3.0;
                if (term1 < orthogonality_threshold &&
                    term2 < orthogonality_threshold &&
                    term3 < orthogonality_threshold) {
                    valid_triplets.push_back({static_cast<int>(i), static_cast<int>(j), static_cast<int>(k), E});
                }
            }
        }
    }

    if (valid_triplets.empty()) return {};
    //质心阈值也要满足
    double min_D = std::numeric_limits<double>::max();
    Triplet best_triplet = {-1, -1, -1, 0};

    for (const auto &t: valid_triplets) {
        Eigen::Vector3f c1 = planes_info[t.i].centroid.head<3>();
        Eigen::Vector3f c2 = planes_info[t.j].centroid.head<3>();
        Eigen::Vector3f c3 = planes_info[t.k].centroid.head<3>();

        double d12 = (c1 - c2).norm();
        double d23 = (c2 - c3).norm();
        double d31 = (c3 - c1).norm();

        double D = std::abs(d12 - target_dist) + std::abs(d23 - target_dist) + std::abs(d31 - target_dist);

        if (D < min_D) {
            min_D = D;
            best_triplet = t;
        }
    }

    if (best_triplet.i == -1) return {};

    std::vector<PlaneInfo> result;
    result.reserve(3);
    result.push_back(planes_info[best_triplet.i]);
    result.push_back(planes_info[best_triplet.j]);
    result.push_back(planes_info[best_triplet.k]);
    return result;
}
//从原始点云中提取出范围内的原始三维标定板
std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> LidarSort::extractFinalPoints(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &raw_cloud,
    const std::vector<PlaneInfo> &selected_planes,
    float dist_threshold,
    double radius) const {
    std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> result_clouds = {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr(new pcl::PointCloud<pcl::PointXYZINormal>()),
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr(new pcl::PointCloud<pcl::PointXYZINormal>()),
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr(new pcl::PointCloud<pcl::PointXYZINormal>())
    };

    if (!raw_cloud || raw_cloud->empty() || selected_planes.size() != 3)
        return result_clouds;

    std::array<Eigen::Vector3f, 3> normals;
    for (size_t i = 0; i < 3; ++i) {
        Eigen::Vector3f n = selected_planes[i].coeffs.head<3>();
        if (n.norm() > 1e-6f)
            n.normalize();
        normals[i] = n;
    }

    for (const auto &pt: raw_cloud->points) {
        int best_plane_idx = -1;
        float min_dist = std::numeric_limits<float>::max();

        for (size_t i = 0; i < selected_planes.size(); ++i) {
            Eigen::Vector3f pt_vec(pt.x, pt.y, pt.z);
            Eigen::Vector3f c = selected_planes[i].centroid.head<3>();
            if ((pt_vec - c).norm() > radius) continue;

            float dist = std::abs(
                selected_planes[i].coeffs[0] * pt.x +
                selected_planes[i].coeffs[1] * pt.y +
                selected_planes[i].coeffs[2] * pt.z +
                selected_planes[i].coeffs[3]
            );

            if (dist < dist_threshold && dist < min_dist) {
                min_dist = dist;
                best_plane_idx = static_cast<int>(i);
            }
        }

        if (best_plane_idx != -1) {
            pcl::PointXYZINormal p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            p.intensity = pt.intensity;
            p.normal_x = normals[best_plane_idx].x();
            p.normal_y = normals[best_plane_idx].y();
            p.normal_z = normals[best_plane_idx].z();
            result_clouds[best_plane_idx]->points.push_back(p);
        }
    }
    return result_clouds;
}
//为原始点云中提取出来的三维标定板排序
std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> LidarSort::sort_planes(
    const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> &final_clouds) const {
    std::array<double, 3> min_z_vals;
    std::array<double, 3> max_y_vals;
    for (int i = 0; i < 3; ++i) {
        double min_z = std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();
        if (final_clouds[i] && !final_clouds[i]->empty()) {
            for (const auto &p: final_clouds[i]->points) {
                if (p.z < min_z) min_z = p.z;
                if (p.y > max_y) max_y = p.y;
            }
        }
        min_z_vals[i] = min_z;
        max_y_vals[i] = max_y;
    }

    int idx3 = 0;
    for (int i = 1; i < 3; ++i) {
        if (min_z_vals[i] < min_z_vals[idx3]) idx3 = i;
    }

    int idx2 = -1;
    for (int i = 0; i < 3; ++i) {
        if (i == idx3) continue;
        if (idx2 < 0 || max_y_vals[i] > max_y_vals[idx2]) idx2 = i;
    }

    int idx1 = -1;
    for (int i = 0; i < 3; ++i) {
        if (i != idx3 && i != idx2) {
            idx1 = i;
            break;
        }
    }

    return {final_clouds[idx1], final_clouds[idx2], final_clouds[idx3]};
}

LidarSort::LidarSort(double leaf_size,
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
          float dist_threshold)
    : leaf_size_(leaf_size),
      min_points_per_plane_(min_points_per_plane),
      normal_k_search_(normal_k_search),
      smoothness_threshold_deg_(smoothness_threshold_deg),
      curvature_threshold_(curvature_threshold),
      ground_smoothness_threshold_deg_(ground_smoothness_threshold_deg),
      ground_curvature_threshold_(ground_curvature_threshold),
      centroid_distance_target_(centroid_distance_target),
      extraction_radius_(extraction_radius),
      plane_orthogonality_threshold_(plane_orthogonality_threshold),
      rg_neighbor_k_(rg_neighbor_k),
      extract_ground_(extract_ground),
      dist_threshold_(dist_threshold) {
    ResetFinalClouds();
}

bool LidarSort::add(const std::string &lidar_name) {
    ResetFinalClouds(); //清空缓存
    all_candidate_planes_.clear();
    pcl::PointCloud<pcl::PointXYZI>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    cout <<endl<< "------------------------------------------------"<< endl;
    cout << "lidar frame: "<< lidar_name << endl;
    if (pcl::io::loadPLYFile<pcl::PointXYZI>(lidar_name, *raw_cloud) != 0) {
        std::cerr << " Failed to load: " << lidar_name << std::endl;
        return false;
    }
    if (raw_cloud->empty())
        return false;
    //降采样
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(raw_cloud);
    vg.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    vg.filter(*cloud_filtered);
    cout<<"lidar_sort: downsampled cloud size: "<< cloud_filtered->size()<<endl;
    if (cloud_filtered->empty())
        return false;
    //提取地面点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    if (extract_ground_) {
        extractGroundPlane(
            cloud_filtered,
            min_points_per_plane_,
            rg_neighbor_k_,
            ground_smoothness_threshold_deg_,
            ground_curvature_threshold_,
            non_ground_cloud);
    } else {
        non_ground_cloud = cloud_filtered;
    }
    
    //对非地面且点云进行区域生长
    std::vector<pcl::PointIndices> obj_clusters = performRegionGrowing(
        non_ground_cloud,
        min_points_per_plane_,
        rg_neighbor_k_,
        smoothness_threshold_deg_,
        curvature_threshold_);
    //从区域生长结果中提取候选平面
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(non_ground_cloud);
    extract.setNegative(false);
    for (const auto &indices: obj_clusters) {
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices(indices));
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
        extract.setIndices(inliers);
        extract.filter(*cloud);
        all_candidate_planes_.push_back(cloud);
    }
    //从候选平面中选择最佳三个平面
    std::vector<PlaneInfo> selected_planes = selectBestThreePlanes(
        all_candidate_planes_, centroid_distance_target_, plane_orthogonality_threshold_);
    if (selected_planes.size() != 3) {
        cout<<"the frame no three planes satisfy plane_orthogonality_threshold"<<endl;
        return false;
    }
    //在原始点云中提取候选平面所对应的范围内点云
    final_clouds_ = extractFinalPoints(raw_cloud, selected_planes, dist_threshold_, extraction_radius_);
    //三个平面点云排序
    final_clouds_ = sort_planes(final_clouds_);
    return true;
}

void LidarSort::GetFinalClouds(std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> &final_clouds) const {
    final_clouds = final_clouds_;
}

void LidarSort::GetCandidatePlanes(std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &candidate_planes) const {
    candidate_planes = all_candidate_planes_;
}


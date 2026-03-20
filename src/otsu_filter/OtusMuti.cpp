#include "OtusMuti.h"

void OtusMuti::ClearFilteredClouds() {
    for (auto &cloud: filtered_clouds_) {
        cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    }
}

bool OtusMuti::ComputeOtsuThreshold(
    const pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr &cloud,
    double &intensity_threshold) {
    if (!cloud || cloud->empty())
        return false;

    double min_intensity = std::numeric_limits<double>::max();
    double max_intensity = -std::numeric_limits<double>::max();
    for (const auto &pt: cloud->points) {
        min_intensity = std::min(min_intensity, static_cast<double>(pt.intensity));
        max_intensity = std::max(max_intensity, static_cast<double>(pt.intensity));
    }

    if (!std::isfinite(min_intensity) || !std::isfinite(max_intensity))
        return false;
    if (max_intensity - min_intensity < 1e-6) {
        intensity_threshold = min_intensity;
        return true;
    }

    const int bins = 256;
    std::vector<double> histogram(bins, 0.0);
    const double range = max_intensity - min_intensity;

    for (const auto &pt: cloud->points) {
        double normalized = (pt.intensity - min_intensity) / range;
        normalized = std::min(1.0, std::max(0.0, normalized));
        int bin = static_cast<int>(std::floor(normalized * (bins - 1)));
        histogram[bin] += 1.0;
    }

    double total_points = static_cast<double>(cloud->size());
    for (double &h: histogram) {
        h /= total_points;
    }

    std::vector<double> cumulative_prob(bins, 0.0);
    std::vector<double> cumulative_mean(bins, 0.0);
    cumulative_prob[0] = histogram[0];
    cumulative_mean[0] = histogram[0] * 0.0;
    for (int i = 1; i < bins; ++i) {
        cumulative_prob[i] = cumulative_prob[i - 1] + histogram[i];
        cumulative_mean[i] = cumulative_mean[i - 1] + histogram[i] * static_cast<double>(i);
    }
    double global_mean = cumulative_mean[bins - 1];

    double best_between_class_var = -1.0;
    int best_threshold_bin = 0;
    for (int t = 0; t < bins; ++t) {
        double prob_bg = cumulative_prob[t];
        double prob_fg = 1.0 - prob_bg;
        if (prob_bg <= 1e-6 || prob_fg <= 1e-6)
            continue;
        double mean_bg = cumulative_mean[t] / prob_bg;
        double mean_fg = (global_mean - cumulative_mean[t]) / prob_fg;
        double between_var = prob_bg * prob_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if (between_var > best_between_class_var) {
            best_between_class_var = between_var;
            best_threshold_bin = t;
        }
    }

    double normalized_threshold = static_cast<double>(best_threshold_bin) / static_cast<double>(bins - 1);
    intensity_threshold = min_intensity + normalized_threshold * range;
    return true;
}

bool OtusMuti::ComputeMultiLevelOtsu(
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &working_cloud,
    double &intensity_threshold,
    const double min_ratio,
    size_t &iterations,
    size_t &removed_points) {
    iterations = 0;
    removed_points = 0;
    while (true) {
        iterations++;
        //进行大津法
        if (!ComputeOtsuThreshold(working_cloud, intensity_threshold)) {
            return false;
        }

        std::vector<int> high_idx;
        std::vector<int> low_idx;
        high_idx.reserve(working_cloud->size());
        low_idx.reserve(working_cloud->size());
        for (size_t i = 0; i < working_cloud->size(); ++i) {
            if (working_cloud->points[i].intensity >= intensity_threshold) {
                high_idx.push_back(static_cast<int>(i));
            } else {
                low_idx.push_back(static_cast<int>(i));
            }
        }

        const size_t total = working_cloud->size();
        if (total == 0) {
            return false;
        }
        const size_t minority = std::min(high_idx.size(), low_idx.size());
        const double minority_ratio = static_cast<double>(minority) / static_cast<double>(total);
        //看看当前次大津法二分结果是否符合比例要求
        if (minority_ratio >= min_ratio) {
            return true;
        }
        //不符合的话就删除小的那一部分，继续大津法
        const std::vector<int> &drop_idx = (high_idx.size() <= low_idx.size()) ? high_idx : low_idx;
        if (drop_idx.empty() || drop_idx.size() >= working_cloud->size()) {
            return false;
        }

        std::vector<char> drop_mask(working_cloud->size(), 0);
        for (int di: drop_idx) {
            if (di >= 0 && di < static_cast<int>(drop_mask.size())) {
                drop_mask[di] = 1;
            }
        }

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZINormal>());
        filtered->reserve(working_cloud->size() - drop_idx.size());

        for (size_t i = 0; i < working_cloud->size(); ++i) {
            if (drop_mask[i])
                continue;
            filtered->push_back(working_cloud->points[i]);
        }

        removed_points += drop_idx.size();
        working_cloud = filtered;

        if (working_cloud->empty()) {
            return false;
        }
    }
}

void OtusMuti::FilterClouds(
    const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> &input_clouds,
    const double min_ratio) {
    for (size_t i = 0; i < filtered_clouds_.size(); ++i) {
        if (!input_clouds[i] || input_clouds[i]->empty()) {
            continue;
        }

        double threshold = 0.0;
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr working_cloud(
            new pcl::PointCloud<pcl::PointXYZINormal>(*input_clouds[i]));
        size_t iter_count = 0;
        size_t removed_points = 0;
        // 每个平面多级大津法
        bool ok = ComputeMultiLevelOtsu(working_cloud, threshold, min_ratio, iter_count, removed_points);
        if (!ok) {
            continue;
        }
        //过滤
        for (const auto &pt: input_clouds[i]->points) {
            if (pt.intensity >= threshold) {
                filtered_clouds_[i]->push_back(pt);
            }
        }
    }
}

OtusMuti::OtusMuti() {
}

void OtusMuti::add(
    const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> &input_clouds,
    double min_ratio) {
    ClearFilteredClouds();//清缓存
    FilterClouds(input_clouds, min_ratio);//多级大津法过滤黑格点云
}

void OtusMuti::GetFilteredClouds(
    std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> &output_clouds) const {
    output_clouds = filtered_clouds_;
}


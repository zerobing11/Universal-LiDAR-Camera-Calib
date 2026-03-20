#ifndef CAMERA_LIDAR_CALIBRATION_CONFIG_HELPER_H
#define CAMERA_LIDAR_CALIBRATION_CONFIG_HELPER_H

#include <string>
#include <vector>

class MSRConfig {
public:
    static std::string data_path;
    static std::string res_path;
    static std::string img_input;
    static std::string lidar_input;
    static double square_len;
    static std::vector<double> origin_corner_uv;
    static std::vector<double> predefined_intrinsic;
    static std::vector<double> predefined_distortion;
    static int checker_row;
    static int checker_col;
    static std::vector<std::vector<int>> lines_plane_pairs;
    static bool otsu;
    static bool extract_ground;
    static bool enable_joint_optimize;
    static bool use_roi;
    static bool wait_enter;
    static double dist_range;

};

class AlgorithmParamConfig {
public:

    static double corner_detect_threshold;
    static double chessboard_threshold;

    static bool wait_enter;
    static double leaf_size;
    static double ground_smoothness_threshold_deg;
    static double ground_curvature_threshold;
    static int min_points_per_plane;
    static int neighbor_k;
    static double smoothness_threshold_deg;
    static double curvature_threshold;
    static double plane_orthogonality_threshold_deg;

    static double ransac_distance_threshold;
};
void LoadMSRConfig(const std::string& yaml_path);
void LoadAlgorithmParam(const std::string& yaml_path);

class ExperimentConfig {
public:
    static std::vector<double> Rcl;
    static std::vector<double> tcl;
};

void LoadExperimentConfig(const std::string& yaml_path);

#endif


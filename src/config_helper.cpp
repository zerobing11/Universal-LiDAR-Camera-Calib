#include "config_helper.h"

#include <iostream>
#include "yaml-cpp/yaml.h"


std::vector<double> ExperimentConfig::Rcl = {};
std::vector<double> ExperimentConfig::tcl = {};

std::string MSRConfig::data_path = "";
std::string MSRConfig::res_path = "";
std::string MSRConfig::img_input = "img";
std::string MSRConfig::lidar_input = "lidar";
double MSRConfig::square_len = 0.2;
std::vector<double> MSRConfig::origin_corner_uv = {};
std::vector<double> MSRConfig::predefined_intrinsic = {};
std::vector<double> MSRConfig::predefined_distortion = {};
int MSRConfig::checker_row = 5;
int MSRConfig::checker_col = 4;
bool MSRConfig::otsu = true;
bool MSRConfig::extract_ground = true;
bool MSRConfig::enable_joint_optimize = true;
bool MSRConfig::use_roi = false;
bool MSRConfig::wait_enter = false;
double MSRConfig::dist_range = 3.4;
std::vector<std::vector<int>> MSRConfig::lines_plane_pairs = {{2, 0}, {1, 0}, {2, 1}};

double AlgorithmParamConfig::corner_detect_threshold = 0.15;
double AlgorithmParamConfig::chessboard_threshold = 0.8;

bool AlgorithmParamConfig::wait_enter = true;
double AlgorithmParamConfig::leaf_size = 0.05;
double AlgorithmParamConfig::ground_smoothness_threshold_deg = 5.0;
double AlgorithmParamConfig::ground_curvature_threshold = 1.0;
int AlgorithmParamConfig::min_points_per_plane = 1000;
int AlgorithmParamConfig::neighbor_k = 30;
double AlgorithmParamConfig::smoothness_threshold_deg = 5.0;
double AlgorithmParamConfig::curvature_threshold = 1.0;
double AlgorithmParamConfig::plane_orthogonality_threshold_deg = 0.35;

double AlgorithmParamConfig::ransac_distance_threshold = 0.05;

void LoadMSRConfig(const std::string& yaml_path)
{
    YAML::Node config;
    try {
        config = YAML::LoadFile(yaml_path);
    } catch (YAML::BadFile &e) {
        std::cout << "MSR yaml read error!" << yaml_path << std::endl;
        exit(1);
    }

    YAML::Node docs_checker = config["Setting"];
    if (docs_checker) {
        MSRConfig::data_path = docs_checker["data_path"].as<std::string>();
        MSRConfig::res_path = docs_checker["res_path"].as<std::string>();
        MSRConfig::img_input = docs_checker["img_input"].as<std::string>();
        MSRConfig::lidar_input = docs_checker["lidar_input"].as<std::string>();
        MSRConfig::square_len = docs_checker["square_len"].as<double>();
        MSRConfig::origin_corner_uv = docs_checker["origin_corner_uv"].as<std::vector<double>>();
        MSRConfig::predefined_intrinsic = docs_checker["predefined_intrinsic"].as<std::vector<double>>();
        MSRConfig::predefined_distortion = docs_checker["predefined_distortion"].as<std::vector<double>>();
        MSRConfig::checker_row = docs_checker["checker_row"].as<int>();
        MSRConfig::checker_col = docs_checker["checker_col"].as<int>();
        if (docs_checker["otsu"]) {
            MSRConfig::otsu = docs_checker["otsu"].as<bool>();
        }
        if (docs_checker["extract_ground"]) {
            MSRConfig::extract_ground = docs_checker["extract_ground"].as<bool>();
        }
        if (docs_checker["enable_joint_optimize"]) {
            MSRConfig::enable_joint_optimize = docs_checker["enable_joint_optimize"].as<bool>();
        }
        if (docs_checker["use_roi"]) {
            MSRConfig::use_roi = docs_checker["use_roi"].as<bool>();
        }
        if (docs_checker["wait_enter"]) {
            MSRConfig::wait_enter = docs_checker["wait_enter"].as<bool>();
        }
        if (docs_checker["dist_range"]) {
            MSRConfig::dist_range = docs_checker["dist_range"].as<double>();
        }
        if (docs_checker["lines_plane_pairs"]) {
            MSRConfig::lines_plane_pairs = docs_checker["lines_plane_pairs"].as<std::vector<std::vector<int>>>();
        }
    }
}

void LoadAlgorithmParam(const std::string& yaml_path)
{
    YAML::Node config;
    try {
        config = YAML::LoadFile(yaml_path);
    } catch (YAML::BadFile &e) {
        std::cout << "AlgorithmParam yaml read error!" << yaml_path << std::endl;
        exit(1);
    }

    YAML::Node sort_camera = config["SortCameraPlanes"];
    if (sort_camera) {
        AlgorithmParamConfig::corner_detect_threshold = sort_camera["corner_detect_threshold"].as<double>();
        AlgorithmParamConfig::chessboard_threshold = sort_camera["chessboard_threshold"].as<double>();
    }

    YAML::Node sort_lidar = config["SortLidarPlanes"];
    if (sort_lidar) {
        AlgorithmParamConfig::wait_enter = sort_lidar["wait_enter"].as<bool>();
        AlgorithmParamConfig::leaf_size = sort_lidar["leaf_size"].as<double>();
        AlgorithmParamConfig::ground_smoothness_threshold_deg = sort_lidar["ground_smoothness_threshold_deg"].as<double>();
        AlgorithmParamConfig::ground_curvature_threshold = sort_lidar["ground_curvature_threshold"].as<double>();
        AlgorithmParamConfig::min_points_per_plane = sort_lidar["min_points_per_plane"].as<int>();
        AlgorithmParamConfig::neighbor_k = sort_lidar["neighbor_k"].as<int>();
        AlgorithmParamConfig::smoothness_threshold_deg = sort_lidar["smoothness_threshold_deg"].as<double>();
        AlgorithmParamConfig::curvature_threshold = sort_lidar["curvature_threshold"].as<double>();
        AlgorithmParamConfig::plane_orthogonality_threshold_deg = sort_lidar["plane_orthogonality_threshold_deg"].as<double>();
    }

    YAML::Node coarse2fine = config["Coarse2Fine"];
    if (coarse2fine) {
        AlgorithmParamConfig::ransac_distance_threshold = coarse2fine["ransac_distance_threshold"].as<double>();
    }
}

void LoadExperimentConfig(const std::string& yaml_path)
{
    YAML::Node config;
    try {
        config = YAML::LoadFile(yaml_path);
    } catch (YAML::BadFile &e) {
        std::cout << "config yaml read error!" << yaml_path << std::endl;
        exit(1);
    }
    YAML::Node experiment_config = config["experiment"] ? config["experiment"] : config;

    ExperimentConfig::Rcl = experiment_config["Rcl"].as<std::vector<double>>();
    ExperimentConfig::tcl = experiment_config["tcl"].as<std::vector<double>>();
}




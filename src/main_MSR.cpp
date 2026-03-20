#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include <ros/package.h>
#include <fstream>
#include <iomanip>

#include "DealString.h"
#include "FileIO.h"
#include "PointCloudUtil.h"
#include "camera/Camera.h"
#include "lidar_sort/LidarSort.h"
#include "otsu_filter/OtusMuti.h"
#include "lidar_corners_detect/LidarCornersDetect.h"
#include "coarse2fine/Coarse2Fine.h"
#include "config_helper.h"
#include "Sqpnp.h"

using namespace std;

int main(int argc, char **argv) {

    ros::init(argc, argv, "camera_calibration");
    ros::NodeHandle n;
    ros::Publisher map_pub = n.advertise<sensor_msgs::PointCloud2>("/MapCloud",1);
    ros::Publisher checker_pub = n.advertise<sensor_msgs::PointCloud2>("/CheckerBoardCloud", 1);
    const std::string yaml_path = ros::package::getPath("camera_calibration") + "/conf/MSRSetting.yaml";
    const std::string algo_yaml_path = ros::package::getPath("camera_calibration") + "/conf/MSRAlgorithmParam.yaml";
    LoadMSRConfig(yaml_path);
    LoadAlgorithmParam(algo_yaml_path);

    // DocsAndChecker
    string data_path = MSRConfig::data_path;
    string path_names = data_path+"/names.txt";
    string extrinsic_path = MSRConfig::res_path;
    string lidar_input = MSRConfig::lidar_input;
    string img_input = MSRConfig::img_input;
    string img_input_dir = data_path + "/" + img_input + "/";
    string lidar_input_dir = data_path + "/" + lidar_input + "/";
    const int checker_corner_row = MSRConfig::checker_row - 1;
    const int checker_corner_col = MSRConfig::checker_col - 1;
    float square_len = static_cast<float>(MSRConfig::square_len);
    bool otsu = MSRConfig::otsu;
    bool extract_ground = MSRConfig::extract_ground;
    bool enable_joint_optimize = MSRConfig::enable_joint_optimize;
    std::pair<double, double> origin_corner_uv = {MSRConfig::origin_corner_uv[0] + square_len, MSRConfig::origin_corner_uv[1] + square_len};
    cv::Mat predefined_intrinsic, predefined_distortion;
    if (MSRConfig::predefined_intrinsic.size() == 9) {
        predefined_intrinsic = (cv::Mat_<double>(3,3) <<
            MSRConfig::predefined_intrinsic[0], MSRConfig::predefined_intrinsic[1], MSRConfig::predefined_intrinsic[2],
            MSRConfig::predefined_intrinsic[3], MSRConfig::predefined_intrinsic[4], MSRConfig::predefined_intrinsic[5],
            MSRConfig::predefined_intrinsic[6], MSRConfig::predefined_intrinsic[7], MSRConfig::predefined_intrinsic[8]);
    }
    if (MSRConfig::predefined_distortion.size() == 5) {
        predefined_distortion = (cv::Mat_<double>(1,5) <<
            MSRConfig::predefined_distortion[0], MSRConfig::predefined_distortion[1],
            MSRConfig::predefined_distortion[2], MSRConfig::predefined_distortion[3],
            MSRConfig::predefined_distortion[4]);
    }
    std::array<std::pair<int, int>, 3> lines_plane_pairs;
    if (MSRConfig::lines_plane_pairs.size() == 3
        && MSRConfig::lines_plane_pairs[0].size() == 2
        && MSRConfig::lines_plane_pairs[1].size() == 2
        && MSRConfig::lines_plane_pairs[2].size() == 2) {
        lines_plane_pairs = {{
            {MSRConfig::lines_plane_pairs[0][0], MSRConfig::lines_plane_pairs[0][1]},
            {MSRConfig::lines_plane_pairs[1][0], MSRConfig::lines_plane_pairs[1][1]},
            {MSRConfig::lines_plane_pairs[2][0], MSRConfig::lines_plane_pairs[2][1]}
        }};
        }

    // SortCameraPlanes
    double corner_detect_threshold = AlgorithmParamConfig::corner_detect_threshold;
    double chessboard_threshold = AlgorithmParamConfig::chessboard_threshold;

    // SortLidarPlanes
    bool wait_enter = AlgorithmParamConfig::wait_enter;
    double leaf_size = AlgorithmParamConfig::leaf_size;
    int min_points_per_plane = AlgorithmParamConfig::min_points_per_plane;
    int neighbor_k = AlgorithmParamConfig::neighbor_k;
    double ground_smoothness_threshold_deg = AlgorithmParamConfig::ground_smoothness_threshold_deg;
    double ground_curvature_threshold = AlgorithmParamConfig::ground_curvature_threshold;
    double smoothness_threshold_deg = AlgorithmParamConfig::smoothness_threshold_deg;
    double curvature_threshold = AlgorithmParamConfig::curvature_threshold;
    double plane_orthogonality_threshold = AlgorithmParamConfig::plane_orthogonality_threshold_deg;    // 3D checkerboard extraction - normal angle threshold
    double diagonal_len = std::sqrt(std::pow((MSRConfig::checker_row + 1) * square_len / 2.0, 2)
        +std::pow((MSRConfig::checker_col + 1) * square_len / 2.0, 2));
    double centroid_distance_target = diagonal_len;   // 3D checkerboard extraction - centroid distance threshold
    double extraction_radius = diagonal_len;    // 3D checkerboard extraction - extraction radius
    cout<<"extraction_radius="<<extraction_radius<<endl;

    // Coarse2Fine
    double ransac_distance_threshold = AlgorithmParamConfig::ransac_distance_threshold;

    // Extract image corners
    map<int,vector<cv::Point2f>> cam_valid2d;
    map<int,vector<cv::Point3f>> cam_valid3d;
    map<int,vector<Eigen::Vector4f>> camera_planes;
    vector<int> camera_valid_frame;
    CAMERA::Camera Cam(checker_corner_col, checker_corner_row, data_path, corner_detect_threshold, chessboard_threshold,
                       square_len, predefined_intrinsic, predefined_distortion,lines_plane_pairs);
    vector<string> lidar_img_names = FileIO::ReadTxt2String(path_names,false);
    for(int i=0;i<lidar_img_names.size();i++)
    {
        vector<string> lidar_img_name = read_format(lidar_img_names[i]," ");
        string img_name=img_input_dir+lidar_img_name[1]+".png";
        bool ischoose=Cam.add(img_name);// Entry function
        if(ischoose)
        {
            camera_valid_frame.push_back(i);
        }
    }
    Cam.Get2Dpoint(cam_valid2d);
    cout<<"corners num: "<<cam_valid2d[0].size()<<endl;
    Cam.Get3Dpoint(cam_valid3d);
    Cam.GetPlanesModels(camera_planes);

    //--------------------Extract lidar corners: sorting, filtering, plane fitting, and corner computation for 3D checkerboard clouds----------------------------
    vector<vector<Point3f>> valid3d(lidar_img_names.size());
    LidarSort lidar_sort(leaf_size,min_points_per_plane,neighbor_k,smoothness_threshold_deg,curvature_threshold,
        centroid_distance_target,extraction_radius,plane_orthogonality_threshold,
        neighbor_k,ground_smoothness_threshold_deg,ground_curvature_threshold,extract_ground,0.05);
    OtusMuti otus_muti;
    LidarCornersDetect lidar_corners_detect(checker_corner_row, checker_corner_col, square_len,origin_corner_uv, lines_plane_pairs);
    Coarse2Fine coarse_2_fine;

    auto init_extrinsic_file = [](const std::string& path) {
        const std::string dir = FileIO::GetParentDir(path);
        if (!dir.empty() && !FileIO::CreateDirectories(dir)) {
            std::cerr << "Failed to create directory for extrinsic file: "
                      << dir << std::endl;
        }
        std::ofstream fout_clear(path, std::ios::trunc);
        if (fout_clear.is_open()) {
            fout_clear << "# Calibration Results" << std::endl;
            fout_clear << "# Format: filename, R11, R12, R13, R21, R22, R23, R31, R32, R33, tx, ty, tz" << std::endl;
            fout_clear << "# ========================================" << std::endl;
            fout_clear.close();
        } else {
            std::cerr << "Failed to open extrinsic file: " << path << std::endl;
        }
    };
    init_extrinsic_file(extrinsic_path);

    auto append_extrinsic = [](const std::string& path,
                               const std::string& frame_name,
                               const Eigen::Matrix3d& Rcl,
                               const Eigen::Vector3d& tcl) {
        std::ofstream fout(path, std::ios::app);
        if (!fout.is_open()) {
            return false;
        }
        fout << frame_name << ", ";
        fout << Rcl(0, 0) << ", " << Rcl(0, 1) << ", " << Rcl(0, 2) << ", ";
        fout << Rcl(1, 0) << ", " << Rcl(1, 1) << ", " << Rcl(1, 2) << ", ";
        fout << Rcl(2, 0) << ", " << Rcl(2, 1) << ", " << Rcl(2, 2) << ", ";
        fout << tcl(0) << ", " << tcl(1) << ", " << tcl(2) << std::endl;
        fout.close();
        return true;
    };

    for (int frame_idx : camera_valid_frame)
    {
        if (frame_idx < 0 || frame_idx >= static_cast<int>(lidar_img_names.size()))
            continue;
        vector<string> lidar_img_name = read_format(lidar_img_names[frame_idx], " ");
        if (lidar_img_name.empty())
            continue;
        string lidar_name = lidar_input_dir + "/" + lidar_img_name[0] + ".ply";


        //--------Sort and extract 3D checkerboard point cloud--------
        bool sort_success = lidar_sort.add(lidar_name);// Sorting/extraction entry
        // //Publish extracted plane results
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> all_candidate_planes;
        lidar_sort.GetCandidatePlanes(all_candidate_planes);
        pcl::PointCloud<pcl::PointXYZI>::Ptr all_candidates_vis(new pcl::PointCloud<pcl::PointXYZI>());
        for(size_t k = 0; k < all_candidate_planes.size(); ++k)
        {
            pcl::PointCloud<pcl::PointXYZI> temp = *all_candidate_planes[k];
            float intensity = (k + 1) * 10.0f;
            for(auto& p : temp.points) p.intensity = intensity;
            *all_candidates_vis += temp;
        }
        if (!all_candidates_vis->empty())
        {
            publishPointCloud(all_candidates_vis, map_pub);
            ros::spinOnce();
            waitForEnter("Candidate planes visualized.", wait_enter);
        }
        if (!sort_success)
            continue;

        std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> final_plane_clouds;
        lidar_sort.GetFinalClouds(final_plane_clouds);
        pcl::PointCloud<pcl::PointXYZI>::Ptr final_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        for (int i = 0; i < final_plane_clouds.size(); i++)
        {
            double class_intensity = i * 40;
            for (const auto& p_n : final_plane_clouds[i]->points)
            {

                pcl::PointXYZI p;
                p.x = p_n.x;
                p.y = p_n.y;
                p.z = p_n.z;
                p.intensity = p_n.intensity;
                // p.intensity = class_intensity;
                final_cloud->points.push_back(p);
            }
        }
        // //Publish final 3D checkerboard
        publishPointCloud(final_cloud, map_pub);
        ros::spinOnce();
        waitForEnter("Press ENTER to otsu(if true) and dropline...", wait_enter);

        // ---------Otsu intensity filtering for checkerboard point cloud----------
        std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> filtered_clouds;
        if (otsu)
        {
            otus_muti.add(final_plane_clouds);// Filtering entry
            otus_muti.GetFilteredClouds(filtered_clouds);
        }
        else {
            filtered_clouds = final_plane_clouds;
        }

        //-------------Plane fitting and inlier retention-----------------
        std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> down_sample_filtered_cloud;
        voxelDownsample(filtered_clouds,0.005,down_sample_filtered_cloud);// Downsample with small voxel size for uniform density
        std::array<Eigen::Vector4f, 3> planes_models;
        planes_models.fill(Eigen::Vector4f::Zero());
        Eigen::Matrix3f R;
        for (size_t i = 0; i < down_sample_filtered_cloud.size(); ++i)
        {
            Eigen::Vector4f plane_model;
            MlesacResult result = RunPlaneMlesac(down_sample_filtered_cloud[i]);
            cout<<"plane-estimation score: "<<result.score<<endl;
            planes_models[i] = result.plane;
        }

        CheckBoardPlane(planes_models);

        //--------------Extract 3D corners----------------
        lidar_corners_detect.add(planes_models);// 3D corner extraction entry
        pcl::PointCloud<pcl::PointXYZI>::Ptr corners = lidar_corners_detect.GetMergedCornersCloud();
        std::vector<std::pair<LineEquation, LineEquation>> line_equations = lidar_corners_detect.GetLineEquations();// Get line equations
        // Prepare PnP corners
        valid3d[frame_idx].reserve(corners->size());
        for (const auto& corner : corners->points) {
            valid3d[frame_idx].push_back(Point3f(corner.x, corner.y, corner.z));
        }
        //--------------Solve coarse extrinsic via PnP----------------
        auto it2d = cam_valid2d.find(frame_idx);
        if (valid3d[frame_idx].size() != it2d->second.size()) {
            std::cout << "frame " << frame_idx << " 2D/3Dcorners num not match: "<< it2d->second.size() << " vs " << valid3d[frame_idx].size() << std::endl;
            continue;
        }
        Eigen::Matrix3d Coarse_Rcl;
        Eigen::Vector3d Coarse_tcl;
        double frame_rms;
        bool ok = SolveSqpnpPnP(valid3d[frame_idx], it2d->second,predefined_intrinsic, predefined_distortion,
                                Coarse_Rcl, Coarse_tcl, frame_rms//3out
                                );
        if (!ok) {
            std::cout << "frame " << frame_idx << " PnP solve failed" << std::endl;
            continue;
        }
        cout<<endl<<"Coarse_Rcl: "<< endl << Coarse_Rcl << endl;
        cout<<"Coarse_tcl: "<< endl<< Coarse_tcl << endl;
        cout<<"Coarse reprojection error RMSE: "<< frame_rms << " pixels" << endl;
        //----------------Filter point clouds by line equations-------------
        std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> down_sample_filtered_clouds_by_lines = FilterCloudByLines(line_equations, down_sample_filtered_cloud, square_len/2,10);
        setPlaneIntensity(down_sample_filtered_clouds_by_lines);
        // Visualization
        pcl::PointCloud<pcl::PointXYZI>::Ptr display_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        for (int i = 0; i < down_sample_filtered_clouds_by_lines.size(); i++)
        {
            for (const auto& p_n : down_sample_filtered_clouds_by_lines[i]->points)
            {
                pcl::PointXYZI p;
                p.x = p_n.x;
                p.y = p_n.y;
                p.z = p_n.z;
                p.intensity = p_n.intensity;
                display_cloud->points.push_back(p);
            }
        }
        // Publish 3D cloud and corners
        publishUntilEnter(
        [&]()
        {
        publishPointCloud(display_cloud, map_pub);
        publishPointCloud(corners, checker_pub);
        },
        "press ENTER to coarse to fine and continue to next frame ...", wait_enter);
        //---------------Coarse-to-fine extrinsic optimization-----------------
        cout<<endl<<"wait for Optimizing"<<endl;
        if (camera_planes.find(frame_idx) != camera_planes.end()) {
             Eigen::Matrix3d Optimized_Rcl;
             Eigen::Vector3d Optimized_tcl;
             coarse_2_fine.add(camera_planes[frame_idx], Coarse_Rcl, Coarse_tcl, filtered_clouds, Optimized_Rcl, Optimized_tcl);
             cout << "Optimized Rcl: " << endl << Optimized_Rcl << endl;
             cout << "Optimized tcl: " << endl << Optimized_tcl.transpose() << endl;

             // Validation: transform camera-frame corners to lidar frame with optimized extrinsic and report reprojection RMSE
             auto it_cam3d = cam_valid3d.find(frame_idx);
             if (it_cam3d != cam_valid3d.end() && it_cam3d->second.size() == it2d->second.size()) {
                 std::vector<cv::Point3f> lidar_corners_from_cam;
                 lidar_corners_from_cam.reserve(it_cam3d->second.size());
                 for (const auto& p_cam_cv : it_cam3d->second) {
                     Eigen::Vector3d p_cam(p_cam_cv.x, p_cam_cv.y, p_cam_cv.z);
                     Eigen::Vector3d p_lidar = Optimized_Rcl.transpose() * (p_cam - Optimized_tcl);
                     lidar_corners_from_cam.emplace_back(
                         static_cast<float>(p_lidar.x()),
                         static_cast<float>(p_lidar.y()),
                         static_cast<float>(p_lidar.z()));
                 }

                 cv::Mat rmat_opt = (cv::Mat_<double>(3, 3) <<
                     Optimized_Rcl(0, 0), Optimized_Rcl(0, 1), Optimized_Rcl(0, 2),
                     Optimized_Rcl(1, 0), Optimized_Rcl(1, 1), Optimized_Rcl(1, 2),
                     Optimized_Rcl(2, 0), Optimized_Rcl(2, 1), Optimized_Rcl(2, 2));
                 cv::Mat rvec_opt, tvec_opt;
                 cv::Rodrigues(rmat_opt, rvec_opt);
                 tvec_opt = (cv::Mat_<double>(3, 1) << Optimized_tcl(0), Optimized_tcl(1), Optimized_tcl(2));

                 std::vector<cv::Point2f> projected_uv_opt;
                 cv::projectPoints(lidar_corners_from_cam, rvec_opt, tvec_opt,
                                   predefined_intrinsic, predefined_distortion, projected_uv_opt);
                 const size_t point_count = std::min(projected_uv_opt.size(), it2d->second.size());
                 double squared_error = 0.0;
                 for (size_t i = 0; i < point_count; ++i) {
                     const double dx = it2d->second[i].x - projected_uv_opt[i].x;
                     const double dy = it2d->second[i].y - projected_uv_opt[i].y;
                     squared_error += dx * dx + dy * dy;
                 }
                 const double rmse_opt = (point_count > 0)
                     ? std::sqrt(squared_error / static_cast<double>(point_count))
                     : 0.0;
                 cout << "Optimized reprojection error RMSE: "<< rmse_opt << " pixels" << endl;
             } 
             
             // Save optimized extrinsic of current frame
             if (!append_extrinsic(extrinsic_path, lidar_img_name[0], Optimized_Rcl, Optimized_tcl)) {
                 std::cerr << "Failed to append to extrinsic file: " << extrinsic_path << std::endl;
             }
        }
    }
    
    // Joint optimization (optional)
    if (enable_joint_optimize) {
        cout << endl << "==================== Joint Optimization ====================" << endl;
        Eigen::Matrix3d Final_Rcl;
        Eigen::Vector3d Final_tcl;
        coarse_2_fine.JointOptimize(Final_Rcl, Final_tcl);
        cout << "Final Joint Optimized Rcl: " << endl << Final_Rcl << endl;
        cout << "Final Joint Optimized tcl: " << endl << Final_tcl.transpose() << endl;
        
        // Save joint-optimized extrinsic to file
        std::ofstream fout(extrinsic_path, std::ios::app);
        if (fout.is_open()) {
            fout << std::endl;
            fout << "# Joint Optimization Result" << std::endl;
            fout << std::fixed << std::setprecision(17);
            fout << "joint, ";
            fout << Final_Rcl(0, 0) << ", " << Final_Rcl(0, 1) << ", " << Final_Rcl(0, 2) << ", ";
            fout << Final_Rcl(1, 0) << ", " << Final_Rcl(1, 1) << ", " << Final_Rcl(1, 2) << ", ";
            fout << Final_Rcl(2, 0) << ", " << Final_Rcl(2, 1) << ", " << Final_Rcl(2, 2) << ", ";
            fout << Final_tcl(0) << ", " << Final_tcl(1) << ", " << Final_tcl(2) << std::endl;
            fout.close();
            cout << "Extrinsic results saved to: " << extrinsic_path << endl;
        } else {
            std::cerr << "Failed to append joint optimization to extrinsic file: " << extrinsic_path << std::endl;
        }
    } else {
        cout << endl << "Joint optimization is disabled." << endl;
    }
}
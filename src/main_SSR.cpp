#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include <ros/package.h>
#include <fstream>

#include "DealString.h"
#include "FileIO.h"
#include "PointCloudUtil.h"
#include "camera/Camera.h"
#include "lidar_sort/LidarSort.h"
#include "otsu_filter/OtusMuti.h"
#include "lidar_corners_detect/LidarCornersDetect.h"
#include "coarse2fine/Coarse2Fine.h"
#include "refine/Refine.h"
#include "config_helper.h"
#include "Sqpnp.h"

using namespace std;

int main(int argc, char **argv) {

    ros::init(argc, argv, "camera_calibration");
    ros::NodeHandle n;
    ros::Publisher map_pub = n.advertise<sensor_msgs::PointCloud2>("/MapCloud",1);
    ros::Publisher checker_pub = n.advertise<sensor_msgs::PointCloud2>("/CheckerBoardCloud", 1);
    const std::string yaml_path = ros::package::getPath("camera_calibration") + "/conf/SSRSetting.yaml";
    const std::string algo_yaml_path = ros::package::getPath("camera_calibration") + "/conf/SSRAlgorithmParam.yaml";
    LoadMSRConfig(yaml_path);
    LoadAlgorithmParam(algo_yaml_path);

    // DocsAndChecker
    string data_path = MSRConfig::data_path;
    string extrinsic_path = MSRConfig::res_path;
    string path_coor = data_path+"/3D.txt";
    string path_names = data_path+"/names.txt";
    string lidar_input = MSRConfig::lidar_input;
    string img_input = MSRConfig::img_input;
    string img_input_dir = data_path + "/" + img_input + "/";
    string lidar_input_dir = data_path + "/" + lidar_input + "/";
    const int checker_corner_row = MSRConfig::checker_row - 1;
    const int checker_corner_col = MSRConfig::checker_col - 1;
    bool otsu = MSRConfig::otsu;
    bool extract_ground = MSRConfig::extract_ground;
    float square_len = static_cast<float>(MSRConfig::square_len);
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
    Cam.Get3Dpoint(cam_valid3d);
    Cam.GetPlanesModels(camera_planes);

    vector<vector<Point3f>> coarse_valid3d(lidar_img_names.size());
    vector<vector<Point3f>> refine_valid3d(lidar_img_names.size());
    LidarSort lidar_sort(leaf_size,min_points_per_plane,neighbor_k,smoothness_threshold_deg,curvature_threshold,
        centroid_distance_target,extraction_radius,plane_orthogonality_threshold,
        neighbor_k,ground_smoothness_threshold_deg,ground_curvature_threshold,extract_ground,0.06);
    OtusMuti otus_muti;
    LidarCornersDetect lidar_corners_detect(checker_corner_row, checker_corner_col, square_len,origin_corner_uv, lines_plane_pairs);
    Coarse2Fine coarse_2_fine;
    Refine refine(checker_corner_row, checker_corner_col, square_len, origin_corner_uv, lines_plane_pairs);

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
        }
    };
    init_extrinsic_file(extrinsic_path);

    auto append_extrinsic = [](const std::string& path,
                               const std::string& name,
                               const Eigen::Matrix3d& Rcl,
                               const Eigen::Vector3d& tcl) {
        std::ofstream fout(path, std::ios::app);
        if (fout.is_open()) {
            fout << name << ", ";
            fout << Rcl(0, 0) << ", " << Rcl(0, 1) << ", " << Rcl(0, 2) << ", ";
            fout << Rcl(1, 0) << ", " << Rcl(1, 1) << ", " << Rcl(1, 2) << ", ";
            fout << Rcl(2, 0) << ", " << Rcl(2, 1) << ", " << Rcl(2, 2) << ", ";
            fout << tcl(0) << ", " << tcl(1) << ", " << tcl(2) << std::endl;
            fout.close();
            return true;
        }
        return false;
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
            for (const auto& p_n : final_plane_clouds[i]->points)
            {
                pcl::PointXYZI p;
                p.x = p_n.x;
                p.y = p_n.y;
                p.z = p_n.z;
                p.intensity = p_n.intensity;
                final_cloud->points.push_back(p);
            }
        }
        // Publish final 3D checkerboard
        publishPointCloud(final_cloud, map_pub);
        ros::spinOnce();
        waitForEnter("Press ENTER to otsu filter...", wait_enter);

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
        // //Publish visualization after intensity filtering
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
        for (const auto& cloud : filtered_clouds) {
            if (cloud && !cloud->empty()) {
                pcl::PointCloud<pcl::PointXYZINormal> temp;
                pcl::copyPointCloud(*cloud, temp);
                *filtered_cloud += temp;
            }
        }
        publishPointCloud(filtered_cloud, map_pub);
        ros::spinOnce();
        waitForEnter("Press ENTER to compute plane_model and corners...", wait_enter);

        //-------------Plane fitting and inlier retention-----------------
        std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> down_sample_filtered_cloud;
        voxelDownsample(filtered_clouds,0.02,down_sample_filtered_cloud);// Downsample with small voxel size for uniform density
        std::array<Eigen::Vector4f, 3> planes_models;
        planes_models.fill(Eigen::Vector4f::Zero());
        for (size_t i = 0; i < down_sample_filtered_cloud.size(); ++i)
        {
            Eigen::Vector4f plane_model;
            MlesacResult result = RunPlaneMlesac(down_sample_filtered_cloud[i]);
            cout<<"plane-estimation score: "<<result.score<<endl;
            planes_models[i] = result.plane;
        }
        CheckBoardPlane(planes_models);

        //---------==-----Extract 3D corners-----------------
        lidar_corners_detect.add(planes_models);// 3D corner extraction entry
        pcl::PointCloud<pcl::PointXYZI>::Ptr coarse_corners = lidar_corners_detect.GetMergedCornersCloud();
        publishPointCloud(coarse_corners, checker_pub);
        std::vector<std::pair<LineEquation, LineEquation>> line_equations = lidar_corners_detect.GetLineEquations();// Get line equations
        // Prepare PnP corners
        coarse_valid3d[frame_idx].reserve(coarse_corners->size());
        for (const auto& corner : coarse_corners->points) {
            coarse_valid3d[frame_idx].push_back(Point3f(corner.x, corner.y, corner.z));
        }
        //--------------Solve coarse extrinsic via PnP----------------
        auto it2d = cam_valid2d.find(frame_idx);
        if (coarse_valid3d[frame_idx].size() != it2d->second.size()) {
            std::cout << "frame " << frame_idx << " 2D/3Dcorners num not match: "<< it2d->second.size() << " vs " << coarse_valid3d[frame_idx].size() << std::endl;
            continue;
        }
        Eigen::Matrix3d Coarse_Rcl;
        Eigen::Vector3d Coarse_tcl;
        double coarse_frame_rms;
        bool ok = SolveSqpnpPnP(coarse_valid3d[frame_idx], it2d->second,predefined_intrinsic, predefined_distortion,
                                Coarse_Rcl, Coarse_tcl, coarse_frame_rms//3out
                                );
        if (!ok) {
            std::cout << "frame " << frame_idx << " PnP solve failed" << std::endl;
            continue;
        }
        cout<<endl<<"Coarse_Rcl: "<< endl << Coarse_Rcl << endl;
        cout<<"Coarse_tcl: "<< endl<< Coarse_tcl.transpose() << endl;
        cout<<"Coarse reprojection error RMSE: "<< coarse_frame_rms << " pixels" << endl;
        //----------------Filter point clouds by line equations-------------
        std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> down_sample_filtered_clouds_by_lines = FilterCloudByLines(line_equations, down_sample_filtered_cloud, square_len/3,0);
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
        publishPointCloud(display_cloud, map_pub);
        ros::spinOnce();
        waitForEnter("Press ENTER to compute plane_model and corners...", wait_enter);
        //---------------Coarse-to-fine extrinsic optimization-----------------
        Eigen::Matrix3d Optimized_Rcl;
        Eigen::Vector3d Optimized_tcl;
        cout<<endl<<"wait for coarse to fine Optimizing"<<endl;
        if (camera_planes.find(frame_idx) != camera_planes.end()) {
             coarse_2_fine.add(camera_planes[frame_idx], Coarse_Rcl, Coarse_tcl, down_sample_filtered_clouds_by_lines, Optimized_Rcl, Optimized_tcl);// Extrinsic optimization entry
             cout << "Optimized Rcl: " << endl << Optimized_Rcl << endl;
             cout << "Optimized tcl: " << endl << Optimized_tcl.transpose() << endl;
        }
        // Transform camera planes to lidar frame with optimized extrinsic
        std::array<Eigen::Vector4f, 3> lidar_planes_opt = coarse_2_fine.GetLidarPlanes();
        auto it3d = cam_valid3d.find(frame_idx);
        pcl::PointCloud<pcl::PointXYZI>::Ptr corners_cam2lidar;
        // Transform camera-frame corners to lidar frame with optimized extrinsic
        transformCam3dToLidar3d(it3d->second,Optimized_Rcl,Optimized_tcl,
            corners_cam2lidar//out
            );
        //-----------------Extrinsic refinement------------------
        cout<<endl<<"wait for fine to refine Optimizing"<<endl;
        std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3> down_sample_final_plane_clouds;
        voxelDownsample(final_plane_clouds,0.005,down_sample_final_plane_clouds);
        refine.add(lidar_planes_opt,corners_cam2lidar,down_sample_final_plane_clouds);// Refine entry function
        pcl::PointCloud<pcl::PointXYZI>::Ptr refine_corners = refine.GetCornersCloudRefine();// Get final refined corners
        // Publish 3D cloud and corners
        publishUntilEnter(
        [&]()
        {
        publishPointCloud(final_cloud, map_pub);
        publishPointCloud(refine_corners, checker_pub);
        },
        "press ENTER to coarse to solve pnp and continue to next frame ...", wait_enter);
        //--------------Final PnP solving----------------------
        // Prepare PnP corners
        refine_valid3d[frame_idx].reserve(refine_corners->size());
        for (const auto& corner : refine_corners->points) {
            refine_valid3d[frame_idx].push_back(Point3f(corner.x, corner.y, corner.z));
        }
        if (refine_valid3d[frame_idx].size() != it2d->second.size()) {
            std::cout << "frame " << frame_idx << " 2D/3Dcorners num not match: "<< it2d->second.size() << " vs " << refine_valid3d[frame_idx].size() << std::endl;
            continue;
        }
        Eigen::Matrix3d Refine_Rcl;
        Eigen::Vector3d Refine_tcl;
        double refine_frame_rms;
        ok = SolveSqpnpPnP(refine_valid3d[frame_idx], it2d->second,predefined_intrinsic, predefined_distortion,
                                Refine_Rcl, Refine_tcl, refine_frame_rms//3out
                                );
        if (!ok) {
            std::cout << "frame " << frame_idx << " PnP solve failed" << std::endl;
            continue;
        }
        cout<<endl<<"Refine_Rcl: "<< endl << Refine_Rcl << endl;
        cout<<"Refine_tcl: "<< endl<< Refine_tcl.transpose() << endl;
        cout<<"Refine reprojection error RMSE: "<< refine_frame_rms << " pixels" << endl;
        if (append_extrinsic(extrinsic_path, lidar_img_name[0], Refine_Rcl, Refine_tcl)) {
            cout << "Extrinsic saved to: " << extrinsic_path << endl;
        } else {
            cerr << "Failed to open file: " << extrinsic_path << endl;
        }
    }
    //----------------Full-batch PnP solving--------------------------
    // Merge 2D and 3D corners from all valid frames
    vector<cv::Point2f> all_2d_points;
    vector<cv::Point3f> all_3d_points;
    for (int frame_idx : camera_valid_frame) {
        auto it2d = cam_valid2d.find(frame_idx);
        if (refine_valid3d[frame_idx].size() != it2d->second.size())
            continue;
        // Merge points
        all_2d_points.insert(all_2d_points.end(), it2d->second.begin(), it2d->second.end());
        all_3d_points.insert(all_3d_points.end(), refine_valid3d[frame_idx].begin(), refine_valid3d[frame_idx].end());
    }

    if (!all_2d_points.empty() && all_2d_points.size() == all_3d_points.size()) {
        cout << endl << "================== Joint PnP ==================" << endl;
        cout << "Total points: " << all_2d_points.size() << endl;

        Eigen::Matrix3d Joint_Rcl;
        Eigen::Vector3d Joint_tcl;
        double joint_rms;
        bool ok = SolveSqpnpPnP(all_3d_points, all_2d_points, predefined_intrinsic, predefined_distortion,
                                Joint_Rcl, Joint_tcl, joint_rms);
        if (ok) {
            cout << "Joint_Rcl: " << endl << Joint_Rcl << endl;
            cout << "Joint_tcl: " << endl << Joint_tcl.transpose() << endl;
            cout << "Joint reprojection error RMSE: " << joint_rms << " pixels" << endl;

            // Save full-batch extrinsic
            if (append_extrinsic(extrinsic_path, "joint", Joint_Rcl, Joint_tcl)) {
                cout << "Joint Extrinsic appended to: " << extrinsic_path << endl;
            }
        }
        else {
            cerr << "Joint PnP solve failed" << endl;
        }
    }
    else {
        cerr << "No valid points for Joint PnP" << endl;
    }

}
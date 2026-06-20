#ifndef _Camera_
#define _Camera_

#include <stdio.h>
#include <string>
#include <fstream>
#include <array>
#include <vector>
#include <map>
#include <Eigen/Eigenvalues>
#include "ChessboradStruct.h"
#include "CornerDetAC.h"

using namespace std;
using namespace cv;

namespace CAMERA
{
    struct BOARD
    {
        vector<cv::Point2f> corners;
        vector<cv::Point3f> corners_3d;
        vector<cv::Point2f> orderd_corners;
        vector<cv::Point3f> orderd_corners_3d;
        Eigen::Vector3f line_u;
        Eigen::Vector3f line_v;
        vector<cv::Point2f> origin_2d; // Origin in the single-board local coordinate frame

        Eigen::Vector3f origin; // 3D calibration board origin in the camera frame
        Eigen::Vector4f plane;
    };

class Camera
{
    private:
        // Chessboard detection parameters
        int numofcorner;
        double corner_detect_threshold;  // Corner detection threshold
        double chessboard_threshold;     // Chessboard recognition threshold

        // Image information
        bool initialization;
        int img_indx;
        cv::Size image_size;
        cv::Mat img,org_img;
        vector<int> camera_cal_frame;

        // Corner information
        int corners_col,corners_row;
        BOARD cur_boards[3];
        map<int,vector<cv::Point2f>> all_2d_corners;
        map<int,vector<cv::Point3f>> all_3d_corners;
		vector<vector<Point3f>> valid3d;
        vector<Point> corners_p;
        Corners corners_s;
        std::vector<cv::Mat> chessboards;

        // Calibration directory
        string path_root;

        // Calibration information
        vector<cv::Mat> rotateMat;
		vector<cv::Mat> translateMat;
        cv::Mat distParameter;
		cv::Mat intrincMatrix;
        Mat mapx;
		Mat mapy;
        map<int,vector<Eigen::Vector4f>> cam_planes;
        std::array<std::pair<int, int>, 3> line_plane_pairs_;

        // Parameters required for PnP-based sorting
        std::vector<cv::Point3f> board_3d_points_;
        cv::Mat intrinsic_for_sort_;
        cv::Mat distortion_for_sort_;
        float square_size_;

    public:
        Camera(int corners_col_,int corners_row_,string path,double corner_thresh,double chess_thresh,
               float square_size, const cv::Mat& intrinsic_matrix, const cv::Mat& distortion_coeffs,std::array<std::pair<int, int>, 3> line_plane_pairs);


        void extract_corners();
        bool sort_boards();
        void compute_line_model(const std::array<std::pair<int, int>, 3>& line_plane_pairs);
        bool compute_plane_angle();
        void sort_corners();
        bool Ensure_ValidFrame(std::vector<cv::Mat> chessboards);
        void visualize_chessboards();
        void visualize_corners();


        void init_img();
        bool add(string path);
        void DataClear();
        void GetIntrincMatrix(cv::Mat &intrincMatrix_);
        void GetDistParameter(cv::Mat &distParameter_);
        void Get2Dpoint(map<int,vector<cv::Point2f>> &all_corners_);
        void Get3Dpoint(map<int,vector<cv::Point3f>> &all_corners_);
        void GetPlanesModels(map<int,vector<Eigen::Vector4f>> &cam_planes_);
};

}

#endif


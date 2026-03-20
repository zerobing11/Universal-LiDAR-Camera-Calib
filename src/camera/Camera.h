#ifndef _Camera_
#define _Camera_

#include <stdio.h>    
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <assert.h>
#include <array>
#include <sstream>
#include <cmath>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>    
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>    
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <Eigen/Dense>
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
        Eigen::Vector3f line_u;//单板基准系u方向
        Eigen::Vector3f line_v;//单板基准系v方向
        vector<cv::Point2f> origin_2d;//单板基准系原点

        Eigen::Vector3f origin;//相机系三维标定板原点
        Eigen::Vector4f plane;
    };

class Camera
{
    private:
        //棋盘检测参数
        int numofcorner;
        double corner_detect_threshold;  // 角点检测阈值
        double chessboard_threshold;     // 棋盘格识别阈值

        //图片信息
        bool initialization;
        int img_indx;
        cv::Size image_size;
        cv::Mat img,org_img;
        vector<int> camera_cal_frame;

        //角点信息
        int corners_col,corners_row;
        BOARD cur_boards[3];
        map<int,vector<cv::Point2f>> all_2d_corners;
        map<int,vector<cv::Point3f>> all_3d_corners;
		vector<vector<Point3f>> valid3d;
        vector<Point> corners_p;
        Corners corners_s;
        std::vector<cv::Mat> chessboards;

        //标定目录
        string path_root;

        //标定信息
        vector<cv::Mat> rotateMat;
		vector<cv::Mat> translateMat;
        cv::Mat distParameter;
		cv::Mat intrincMatrix;
        Mat mapx;
		Mat mapy;
        map<int,vector<Eigen::Vector4f>> cam_planes;
        std::array<std::pair<int, int>, 3> line_plane_pairs_;

        // PnP排序所需的参数
        std::vector<cv::Point3f> board_3d_points_;  // 单个棋盘格系下的角点坐标
        cv::Mat intrinsic_for_sort_;
        cv::Mat distortion_for_sort_;
        float square_size_;

    public:
        Camera(int corners_col_,int corners_row_,string path,double corner_thresh,double chess_thresh,
               float square_size, const cv::Mat& intrinsic_matrix, const cv::Mat& distortion_coeffs,std::array<std::pair<int, int>, 3> line_plane_pairs);


        void extract_corners();//提取棋盘格角点坐标,确保行列顺序正确
        bool sort_boards();//板子排序，返回三平面夹角是否满足阈值
        void compute_line_model(const std::array<std::pair<int, int>, 3>& line_plane_pairs);  // 计算三个平面的交线方向向量
        bool compute_plane_angle();  // 计算三块平面两两之间的法向夹角并判断是否都大于80度
        void sort_corners();  // 对角点进行排序，生成 orderd_corners 和 orderd_corners_3d
        bool Ensure_ValidFrame(std::vector<cv::Mat> chessboards);
        void visualize_chessboards();  // 按排序后的顺序可视化棋盘格
        void visualize_corners();   //可视化角点排序结果
        void visualize_masks();     // 根据角点矩形区域生成遮罩并保存


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


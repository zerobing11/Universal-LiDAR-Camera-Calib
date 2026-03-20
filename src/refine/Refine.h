#ifndef _REFINE_H_
#define _REFINE_H_

#include <vector>
#include <array>
#include <utility>
#include <memory>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/core/types.hpp>
#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <ceres/rotation.h>

#include "PointCloudUtil.h"

// LM优化结果
struct LMResult {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    double k = 1.0;
    double final_cost = 0.0;
    int iterations = 0;
};
//理想棋盘格模型
struct PerfectCheckerBoard {
    PlaneBasis basis;//基准坐标系
    double rect = 0.0;//棋盘格边长
    // std::vector<std::vector<pcl::PointXYZI>> corners_cam2lidar;//通过细化外参从相机系转来的雷达系下棋盘格角点坐标
    std::vector<std::vector<pcl::PointXYZI>> corners;//雷达系下棋盘格角点坐标
    std::vector<std::vector<pcl::PointXYZI>> corners_growth;//标准棋盘格角点坐标
    // std::vector<std::vector<Eigen::Vector2d>> corners_uv;//单板基准系下棋盘格角点坐标
    // std::vector<std::vector<pcl::PointXYZI>> cell_centroids;//雷达系下格子中心点
    std::vector<std::vector<pcl::PointXYZI>> cell_centroids_growth;//雷达系下格子中心点
    // std::vector<std::vector<Eigen::Vector2d>> cell_centroids_uv;//完美棋盘格系下格子中心点
    std::vector<std::vector<double>> cell_centroids_color;

    void resize(int rows, int cols) {
        // corners_cam2lidar.assign(rows, std::vector<pcl::PointXYZI>(cols));
        corners.assign(rows, std::vector<pcl::PointXYZI>(cols));
        corners_growth.assign(rows + 2, std::vector<pcl::PointXYZI>(cols + 2));
        // corners_uv.assign(rows, std::vector<Eigen::Vector2d>(cols));
        
        if (rows > 1 && cols > 1) {
            // cell_centroids.assign(rows - 1, std::vector<pcl::PointXYZI>(cols - 1));
            // cell_centroids_uv.assign(rows - 1, std::vector<Eigen::Vector2d>(cols - 1));
        }
        cell_centroids_growth.assign(rows + 1, std::vector<pcl::PointXYZI>(cols + 1));
        cell_centroids_color.assign(rows + 1, std::vector<double>(cols + 1));
    }
};

class Refine {
private:
    //一些标定板输入参数
    int rows_;
    int cols_;
    double square_len_;
    std::array<PerfectCheckerBoard,3> perfect_checkerboards;
    std::pair<double, double> origin_corner_uv_;//最接近三维标定板交点的那个格子角点相对于三维标定板交点的坐标
    std::array<std::pair<int, int>, 3> lines_plane_pairs_;

    //输出参数
    std::vector<std::pair<LineEquation, LineEquation>> line_equations_;//planei的两条交线
    LMResult lm_result_;//优化结果
    std::array<Eigen::Vector4f, 3> planes_refine_;//优化后的平面方程
    pcl::PointCloud<pcl::PointXYZI>::Ptr corners_cloud_refine_;//优化后平面方程上的角点点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr centroids_cloud_refine_;//优化后平面方程上的质心点云


    //计算三平面交线交点
    void ComputePlaneIntersectionInfo(const std::array<Eigen::Vector4f, 3>& planes,
                                           const std::array<std::pair<int, int>, 3>& lines_plane_pairs,
                                           Eigen::Vector3d& intersection_point//out
                                           );
    //建立平面基准坐标系
    static void buildPlaneBasis(const Eigen::Vector4f& plane,
                         const LineEquation& line_i,
                         const LineEquation& line_j,
                         const Eigen::Vector3d& intersection_point,
                         PlaneBasis& basis//out
                         );
    bool splitCorners(const pcl::PointCloud<pcl::PointXYZI>::Ptr& corners_all,
                      std::vector<std::vector<pcl::PointXYZI>>& corners_split);
     //uv系转雷达系
    void transformCheckerBoardToLidarFrame(const PlaneBasis& basis, const std::vector<std::vector<Eigen::Vector2d>>& corners_uv, std::vector<std::vector<pcl::PointXYZI>>& corners);
    //对齐角点
    void alignCorners(PlaneBasis& basis,
                      const std::vector<pcl::PointXYZI>& corners_board,
                      std::vector<std::vector<pcl::PointXYZI>>& corners);
    //角点生长
    void growCorners(PerfectCheckerBoard& board);
    //计算生长后的质心
    void computeCellCentroids(PerfectCheckerBoard& board);
    //计算格子颜色
    void computeCellColors(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, PerfectCheckerBoard& board);
    //计算入射角与光斑半径并写入点云，并过滤入射角超阈值点
    void computeAngleRadiusAndFilter(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud,
                                   const Eigen::Vector4f& plane,
                                   double K_deg,//平均发散角
                                   double angle_threshold//入射角阈值
                                );
    //优化位姿
    LMResult OptimizePlanePose(const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& clouds,
                               const std::array<PerfectCheckerBoard, 3>& boards,
                               const Eigen::Matrix3d& R0 = Eigen::Matrix3d::Identity(),
                               const Eigen::Vector3d& t0 = Eigen::Vector3d::Zero());
    //逆变换平面方程和角点质心点云
    void computeInverseTransform(const std::array<Eigen::Vector4f, 3>& planes);

public:
    Refine(int rows, int cols, double square_len, const std::pair<double, double>& origin_corner_uv, const std::array<std::pair<int, int>, 3>& lines_plane_pairs);

    void add(const std::array<Eigen::Vector4f, 3>& planes,
             const pcl::PointCloud<pcl::PointXYZI>::Ptr& corners_cam2lidar,
             std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& clouds);

    const std::array<PerfectCheckerBoard, 3>& GetPerfectCheckerboards() const;

    const std::vector<std::pair<LineEquation, LineEquation>>& GetLineEquations() const;

    //获取角点点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr GetCornersCloud() const;
    //获取质心点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr GetCentroidsCloud() const;

    const LMResult& GetLMResult() const;
    //获取refine后的平面方程
    const std::array<Eigen::Vector4f, 3>& GetPlanesRefine() const;
    //获取refine后的角点点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr GetCornersCloudRefine() const;
    //获取refine的质心点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr GetCentroidsCloudRefine() const;
};

#endif


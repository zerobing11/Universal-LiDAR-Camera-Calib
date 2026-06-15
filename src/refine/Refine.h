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
    double k;
    double final_cost;
    int iterations;
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

// 旋转向量转旋转矩阵
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& w) {
    double theta = w.norm();
    if (theta < 1e-9)
        return Eigen::Matrix3d::Identity();
    Eigen::Vector3d n = w / theta;
    Eigen::Matrix3d K;
    K << 0, -n(2), n(1),
         n(2), 0, -n(0),
        -n(1), n(0), 0;
    return Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1 - std::cos(theta)) * K * K;
}

double kTwoOverSqrtPi = 1.12837916709551257390;
double CeresErf(double x) {
    return std::erf(x);
}
template <typename T, int N>
ceres::Jet<T, N> CeresErf(const ceres::Jet<T, N>& x) {
    ceres::Jet<T, N> result;
    //erf直接计算实部
    result.a = std::erf(x.a);
    //计算导数
    const double derivative =
        kTwoOverSqrtPi * std::exp(-x.a * x.a);
    result.v = x.v * derivative;
    return result;
}

double GetScalar(double x) { return x; }
template<typename T, int N>
double GetScalar(const ceres::Jet<T, N>& x) { return x.a; }

// 计算完美棋盘格上，点的理论强度
template <typename T>
T CalculatePerfectIntensity(const T& u, const T& v, const T& intensity_world, const T& radius, const PerfectCheckerBoard& board)
{
    /*
     *即使发生了出格子的情况，强度值也不会发生突变！！！
    */
    double u_scalar = GetScalar(u);
    double v_scalar = GetScalar(v);
    // double intensity_world_scalar = GetScalar(intensity_world);
    double rect = board.rect;

    // 找到最近的垂直线索引和水平线索引
    int col_idx = std::round(u_scalar / rect);
    int row_idx = std::round(v_scalar / rect);
    //我们只关注内棋盘格，不在内棋盘格内的都返回本身强度
    int num_rows = static_cast<int>(board.cell_centroids_color.size());
    int num_cols = static_cast<int>(board.cell_centroids_color[0].size());
    if (col_idx < 1 || col_idx > num_cols - 1 ||
        row_idx < 1 || row_idx > num_rows - 1) {
        return intensity_world;
    }

    // 获取 2x2 格子区域的格子平均强度
    const auto& cells = board.cell_centroids_color;
    double I_BL = cells[row_idx - 1][col_idx - 1];
    double I_BR = cells[row_idx - 1][col_idx];
    double I_TL = cells[row_idx][col_idx - 1];
    double I_TR = cells[row_idx][col_idx];

    // ------------下面就是带导数的部分了------------
    // 计算点相对于十字中心的偏移
    T center_u = T(col_idx * rect);
    T center_v = T(row_idx * rect);
    T du = u - center_u;
    T dv = v - center_v;

    // 高斯参数
    T sigma = radius / T(2); // 假设半径是2sigma
    T factor = T(1.0) / (sigma * T(1.41421356));
    //计算四个方向上的权重
    T erf_u = CeresErf(du * factor);
    T erf_v = CeresErf(dv * factor);
    T w_right = T(0.5) * (T(1.0) + erf_u);
    T w_left  = T(0.5) * (T(1.0) - erf_u);
    T w_top    = T(0.5) * (T(1.0) + erf_v);
    T w_bottom = T(0.5) * (T(1.0) - erf_v);
    //四象限加权求和
    T intensity =
        w_left  * w_bottom * T(I_BL) +
        w_right * w_bottom * T(I_BR) +
        w_left  * w_top    * T(I_TL) +
        w_right * w_top    * T(I_TR);

    return intensity;
}

// 优化的 Cost Functor
struct CheckerboardCostFunctor {
    // 构造函数传入观测数据和模型数据
    CheckerboardCostFunctor(const Eigen::Vector3d& point,
                            double intensity_world,
                            double radius,
                            const PerfectCheckerBoard* board)
        : point_(point), intensity_world_(intensity_world), radius_(radius), board_(board)
    {
        origin_ = board_->basis.origin.cast<double>();
        u_axis_ = board_->basis.u.cast<double>();
        v_axis_ = board_->basis.v.cast<double>();
        plane_ = board_->basis.plane.cast<double>();
    }

    template <typename T>
    bool operator()(const T* const rot_vec, const T* const trans, const T* const k_param, T* residual) const {
        // 1、坐标转换：将最新点云从当前坐标系变换到完美棋盘格坐标系
        T p_raw[3] = {T(point_[0]), T(point_[1]), T(point_[2])};
        T p_trans[3];
        // 将点云从原始雷达系转到最新雷达系
        ceres::AngleAxisRotatePoint(rot_vec, p_raw, p_trans);
        p_trans[0] += trans[0];
        p_trans[1] += trans[1];
        p_trans[2] += trans[2];
        // 射线起点应为t(因为点云平移了 t，雷达原点在当前系下也是 t)
        // 射线方向为 dir = p_trans - t，射线方程为 p(alpha) = t + alpha * dir
        T dir[3];
        dir[0] = p_trans[0] - trans[0];
        dir[1] = p_trans[1] - trans[1];
        dir[2] = p_trans[2] - trans[2];
        T plane_n[3] = {T(plane_[0]), T(plane_[1]), T(plane_[2])};
        T plane_d = T(plane_[3]);
        // 求射线方程p(alpha)与平面方程和的交点，即将射线方程带入平面方程，得到alpha = -(n*t + d) / (n*dir)
        T numer = -(plane_n[0] * trans[0] + plane_n[1] * trans[1] + plane_n[2] * trans[2] + plane_d);
        T denom = plane_n[0] * dir[0] + plane_n[1] * dir[1] + plane_n[2] * dir[2];
        if (ceres::abs(denom) < T(1e-7)) {
            residual[0] = T(0.0);
            return true;
        }
        T alpha = numer / denom;
        T p_proj[3];
        // 将alpha代入射线方程，得到投影点p_proj
        p_proj[0] = trans[0] + alpha * dir[0];
        p_proj[1] = trans[1] + alpha * dir[1];
        p_proj[2] = trans[2] + alpha * dir[2];

        // 将投影点转换到完美棋盘格平面的坐标系
        T vec_diff[3];
        vec_diff[0] = p_proj[0] - T(origin_[0]);
        vec_diff[1] = p_proj[1] - T(origin_[1]);
        vec_diff[2] = p_proj[2] - T(origin_[2]);
        T u_axis_T[3] = {T(u_axis_[0]), T(u_axis_[1]), T(u_axis_[2])};
        T v_axis_T[3] = {T(v_axis_[0]), T(v_axis_[1]), T(v_axis_[2])};
        T u_val = vec_diff[0] * u_axis_T[0] + vec_diff[1] * u_axis_T[1] + vec_diff[2] * u_axis_T[2];
        T v_val = vec_diff[0] * v_axis_T[0] + vec_diff[1] * v_axis_T[1] + vec_diff[2] * v_axis_T[2];
        // 2、计算完美棋盘格上的理论比值
        T radius_val = T(radius_) * k_param[0];
        T radio_perfect = CalculatePerfectIntensity(u_val, v_val, T(intensity_world_), radius_val, *board_);
        // 3、计算残差
        residual[0] = radio_perfect - T(intensity_world_);
        return true;
    }

    // 观测数据
    Eigen::Vector3d point_;
    double intensity_world_;
    double radius_;

    // 棋盘格信息
    const PerfectCheckerBoard* board_;

    // 几何信息
    Eigen::Vector3d origin_;
    Eigen::Vector3d u_axis_;
    Eigen::Vector3d v_axis_;
    Eigen::Vector4d plane_;
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


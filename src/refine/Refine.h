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

struct LMResult {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    double k;
    double final_cost;
    int iterations;
};
// Ideal checkerboard model.
struct PerfectCheckerBoard {
    PlaneBasis basis; // Reference coordinate frame.
    double rect = 0.0;
    std::vector<std::vector<pcl::PointXYZI>> corners;
    std::vector<std::vector<pcl::PointXYZI>> corners_growth;
    std::vector<std::vector<pcl::PointXYZI>> cell_centroids_growth;
    std::vector<std::vector<double>> cell_centroids_color;

    void resize(int rows, int cols) {
        corners.assign(rows, std::vector<pcl::PointXYZI>(cols));
        corners_growth.assign(rows + 2, std::vector<pcl::PointXYZI>(cols + 2));
        cell_centroids_growth.assign(rows + 1, std::vector<pcl::PointXYZI>(cols + 1));
        cell_centroids_color.assign(rows + 1, std::vector<double>(cols + 1));
    }
};

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
    result.a = std::erf(x.a);
    const double derivative =
        kTwoOverSqrtPi * std::exp(-x.a * x.a);
    result.v = x.v * derivative;
    return result;
}

double GetScalar(double x) { return x; }
template<typename T, int N>
double GetScalar(const ceres::Jet<T, N>& x) { return x.a; }

// Compute the ideal intensity of a point on the perfect checkerboard.
template <typename T>
T CalculatePerfectIntensity(const T& u, const T& v, const T& intensity_world, const T& radius, const PerfectCheckerBoard& board)
{

    double u_scalar = GetScalar(u);
    double v_scalar = GetScalar(v);
    double rect = board.rect;

    int col_idx = std::round(u_scalar / rect);
    int row_idx = std::round(v_scalar / rect);
    // Only care about inner checkerboard area; return original intensity if outside.
    int num_rows = static_cast<int>(board.cell_centroids_color.size());
    int num_cols = static_cast<int>(board.cell_centroids_color[0].size());
    if (col_idx < 1 || col_idx > num_cols - 1 ||
        row_idx < 1 || row_idx > num_rows - 1) {
        return intensity_world;
    }

    const auto& cells = board.cell_centroids_color;
    double I_BL = cells[row_idx - 1][col_idx - 1];
    double I_BR = cells[row_idx - 1][col_idx];
    double I_TL = cells[row_idx][col_idx - 1];
    double I_TR = cells[row_idx][col_idx];

    // Compute the offset of the point relative to the cross center.
    T center_u = T(col_idx * rect);
    T center_v = T(row_idx * rect);
    T du = u - center_u;
    T dv = v - center_v;

    // Gaussian parameters.
    T sigma = radius / T(2);
    T factor = T(1.0) / (sigma * T(1.41421356));

    // Compute weights along four directions.
    T erf_u = CeresErf(du * factor);
    T erf_v = CeresErf(dv * factor);
    T w_right = T(0.5) * (T(1.0) + erf_u);
    T w_left  = T(0.5) * (T(1.0) - erf_u);
    T w_top    = T(0.5) * (T(1.0) + erf_v);
    T w_bottom = T(0.5) * (T(1.0) - erf_v);

    // Weighted sum over the four quadrants.
    T intensity =
        w_left  * w_bottom * T(I_BL) +
        w_right * w_bottom * T(I_BR) +
        w_left  * w_top    * T(I_TL) +
        w_right * w_top    * T(I_TR);

    return intensity;
}

struct CheckerboardCostFunctor {
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
        // 1) Coordinate transform: move the current point from lidar frame to the perfect checkerboard frame.
        T p_raw[3] = {T(point_[0]), T(point_[1]), T(point_[2])};
        T p_trans[3];
        ceres::AngleAxisRotatePoint(rot_vec, p_raw, p_trans);
        p_trans[0] += trans[0];
        p_trans[1] += trans[1];
        p_trans[2] += trans[2];

        T dir[3];
        dir[0] = p_trans[0] - trans[0];
        dir[1] = p_trans[1] - trans[1];
        dir[2] = p_trans[2] - trans[2];
        T plane_n[3] = {T(plane_[0]), T(plane_[1]), T(plane_[2])};
        T plane_d = T(plane_[3]);

        T numer = -(plane_n[0] * trans[0] + plane_n[1] * trans[1] + plane_n[2] * trans[2] + plane_d);
        T denom = plane_n[0] * dir[0] + plane_n[1] * dir[1] + plane_n[2] * dir[2];
        if (ceres::abs(denom) < T(1e-7)) {
            residual[0] = T(0.0);
            return true;
        }
        T alpha = numer / denom;
        T p_proj[3];
        p_proj[0] = trans[0] + alpha * dir[0];
        p_proj[1] = trans[1] + alpha * dir[1];
        p_proj[2] = trans[2] + alpha * dir[2];

        T vec_diff[3];
        vec_diff[0] = p_proj[0] - T(origin_[0]);
        vec_diff[1] = p_proj[1] - T(origin_[1]);
        vec_diff[2] = p_proj[2] - T(origin_[2]);
        T u_axis_T[3] = {T(u_axis_[0]), T(u_axis_[1]), T(u_axis_[2])};
        T v_axis_T[3] = {T(v_axis_[0]), T(v_axis_[1]), T(v_axis_[2])};
        T u_val = vec_diff[0] * u_axis_T[0] + vec_diff[1] * u_axis_T[1] + vec_diff[2] * u_axis_T[2];
        T v_val = vec_diff[0] * v_axis_T[0] + vec_diff[1] * v_axis_T[1] + vec_diff[2] * v_axis_T[2];
        // 2) Compute the theoretical intensity on the perfect checkerboard.
        T radius_val = T(radius_) * k_param[0];
        T radio_perfect = CalculatePerfectIntensity(u_val, v_val, T(intensity_world_), radius_val, *board_);
        // 3) Compute the residual.
        residual[0] = radio_perfect - T(intensity_world_);
        return true;
    }

    Eigen::Vector3d point_;
    double intensity_world_;
    double radius_;

    const PerfectCheckerBoard* board_;

    Eigen::Vector3d origin_;
    Eigen::Vector3d u_axis_;
    Eigen::Vector3d v_axis_;
    Eigen::Vector4d plane_;
};

class Refine {
private:
    int rows_;
    int cols_;
    double square_len_;
    std::array<PerfectCheckerBoard,3> perfect_checkerboards;
    std::pair<double, double> origin_corner_uv_;
    std::array<std::pair<int, int>, 3> lines_plane_pairs_;

    std::vector<std::pair<LineEquation, LineEquation>> line_equations_;
    LMResult lm_result_;
    std::array<Eigen::Vector4f, 3> planes_refine_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr corners_cloud_refine_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr centroids_cloud_refine_;

    void ComputePlaneIntersectionInfo(const std::array<Eigen::Vector4f, 3>& planes,
                                           const std::array<std::pair<int, int>, 3>& lines_plane_pairs,
                                           Eigen::Vector3d& intersection_point//out
                                           );
    static void buildPlaneBasis(const Eigen::Vector4f& plane,
                         const LineEquation& line_i,
                         const LineEquation& line_j,
                         const Eigen::Vector3d& intersection_point,
                         PlaneBasis& basis//out
                         );
    bool splitCorners(const pcl::PointCloud<pcl::PointXYZI>::Ptr& corners_all,
                      std::vector<std::vector<pcl::PointXYZI>>& corners_split);

    void transformCheckerBoardToLidarFrame(const PlaneBasis& basis, const std::vector<std::vector<Eigen::Vector2d>>& corners_uv, std::vector<std::vector<pcl::PointXYZI>>& corners);
    void alignCorners(PlaneBasis& basis,
                      const std::vector<pcl::PointXYZI>& corners_board,
                      std::vector<std::vector<pcl::PointXYZI>>& corners);
    void growCorners(PerfectCheckerBoard& board);
    void computeCellCentroids(PerfectCheckerBoard& board);
    void computeCellColors(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, PerfectCheckerBoard& board);
    void computeAngleRadiusAndFilter(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud,
                                   const Eigen::Vector4f& plane,
                                   double K_deg,
                                   double angle_threshold
                                );
    LMResult OptimizePlanePose(const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& clouds,
                               const std::array<PerfectCheckerBoard, 3>& boards,
                               const Eigen::Matrix3d& R0 = Eigen::Matrix3d::Identity(),
                               const Eigen::Vector3d& t0 = Eigen::Vector3d::Zero());
    void computeInverseTransform(const std::array<Eigen::Vector4f, 3>& planes);

public:
    Refine(int rows, int cols, double square_len, const std::pair<double, double>& origin_corner_uv, const std::array<std::pair<int, int>, 3>& lines_plane_pairs);

    void add(const std::array<Eigen::Vector4f, 3>& planes,
             const pcl::PointCloud<pcl::PointXYZI>::Ptr& corners_cam2lidar,
             std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& clouds);

    const std::array<PerfectCheckerBoard, 3>& GetPerfectCheckerboards() const;

    const std::vector<std::pair<LineEquation, LineEquation>>& GetLineEquations() const;

    pcl::PointCloud<pcl::PointXYZI>::Ptr GetCornersCloud() const;
    pcl::PointCloud<pcl::PointXYZI>::Ptr GetCentroidsCloud() const;

    const LMResult& GetLMResult() const;
    const std::array<Eigen::Vector4f, 3>& GetPlanesRefine() const;
    pcl::PointCloud<pcl::PointXYZI>::Ptr GetCornersCloudRefine() const;
    pcl::PointCloud<pcl::PointXYZI>::Ptr GetCentroidsCloudRefine() const;
};

#endif


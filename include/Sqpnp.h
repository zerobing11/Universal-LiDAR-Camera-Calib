#ifndef SQPNPH
#define SQPNPH

#include <vector>

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <sqpnp.h>

bool SolveSqpnpPnP(const std::vector<cv::Point3f>& object_points,
                          const std::vector<cv::Point2f>& image_points,
                          const cv::Mat& intrinsic,
                          const cv::Mat& distortion,
                          Eigen::Matrix3d& Rcl,
                          Eigen::Vector3d& tcl,
                          double& reprojection_rmse) {
    reprojection_rmse = 0.0;
    if (object_points.empty() || image_points.empty()) {
        return false;
    }

    std::vector<cv::Point2f> undistorted_points;
    cv::undistortPoints(image_points, undistorted_points, intrinsic, distortion);

    std::vector<sqpnp::_Point> sqpnp_points;
    std::vector<sqpnp::_Projection> sqpnp_projections;
    sqpnp_points.reserve(object_points.size());
    sqpnp_projections.reserve(undistorted_points.size());

    for (size_t j = 0; j < object_points.size() && j < undistorted_points.size(); ++j) {
        const cv::Point3f& p3 = object_points[j];
        const cv::Point2f& p2 = undistorted_points[j];
        sqpnp_points.emplace_back(p3.x, p3.y, p3.z);
        sqpnp_projections.emplace_back(p2.x, p2.y);
    }

    if (sqpnp_points.empty() || sqpnp_projections.empty()) {
        return false;
    }

    sqpnp::SolverParameters params;
    params.omega_nullspace_method = sqpnp::OmegaNullspaceMethod::RRQR;
    std::vector<double> weights(sqpnp_points.size(), 1.0);
    sqpnp::PnPSolver solver(sqpnp_points, sqpnp_projections, weights, params);

    bool ok = false;
    sqpnp::SQPSolution solution;
    if (solver.IsValid()) {
        solver.Solve();
        const sqpnp::SQPSolution* sol = solver.SolutionPtr(0);
        if (sol != nullptr) {
            solution = *sol;
            ok = true;
        }
    }

    if (!ok) {
        return false;
    }

    Rcl << solution.r_hat[0], solution.r_hat[1], solution.r_hat[2],
           solution.r_hat[3], solution.r_hat[4], solution.r_hat[5],
           solution.r_hat[6], solution.r_hat[7], solution.r_hat[8];
    tcl << solution.t[0], solution.t[1], solution.t[2];

    cv::Mat rmat = (cv::Mat_<double>(3, 3) <<
                    Rcl(0, 0), Rcl(0, 1), Rcl(0, 2),
                    Rcl(1, 0), Rcl(1, 1), Rcl(1, 2),
                    Rcl(2, 0), Rcl(2, 1), Rcl(2, 2));
    cv::Mat rvec, tvec;
    cv::Rodrigues(rmat, rvec);
    tvec = (cv::Mat_<double>(3, 1) << tcl(0), tcl(1), tcl(2));

    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points, rvec, tvec, intrinsic, distortion, projected_points);

    size_t point_count = std::min(projected_points.size(), image_points.size());
    if (point_count == 0) {
        return false;
    }

    double frame_squared_error = 0.0;
    for (size_t j = 0; j < point_count; ++j) {
        double dx = image_points[j].x - projected_points[j].x;
        double dy = image_points[j].y - projected_points[j].y;
        frame_squared_error += dx * dx + dy * dy;
    }
    reprojection_rmse = std::sqrt(frame_squared_error / static_cast<double>(point_count));

    return true;
}

#endif


#include "Coarse2Fine.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <iostream>


Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& w) {
    double theta = w.norm();
    if (theta < 1e-9) return Eigen::Matrix3d::Identity();
    Eigen::Vector3d n = w / theta;
    Eigen::Matrix3d K;
    K << 0, -n(2), n(1), n(2), 0, -n(0), -n(1), n(0), 0;
    return Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1 - std::cos(theta)) * K * K;
}

// Cost for a single point.
struct PlaneCostFunctor {
    PlaneCostFunctor(const Eigen::Vector3d& point,
                  const Eigen::Vector4d& plane)
        : point_(point), plane_(plane) {}

    template <typename T>
    bool operator()(const T* const rot_vec, const T* const trans, T* residual) const {
        T p_rotated[3];
        T point_T[3] = {T(point_[0]), T(point_[1]), T(point_[2])};
        // Rotate the point from lidar frame to camera frame.
        ceres::AngleAxisRotatePoint(rot_vec, point_T, p_rotated);
        T Xc[3] = {p_rotated[0] + trans[0], p_rotated[1] + trans[1], p_rotated[2] + trans[2]};

        // Compute the distance to the corresponding plane.
        residual[0] = T(plane_[0]) * Xc[0] + T(plane_[1]) * Xc[1] + T(plane_[2]) * Xc[2] + T(plane_[3]);
        return true;
    }
    Eigen::Vector3d point_;
    Eigen::Vector4d plane_;
};

Coarse2Fine::Coarse2Fine() {}

Coarse2Fine::~Coarse2Fine() {}

bool Coarse2Fine::add(const std::vector<Eigen::Vector4f>& camera_planes,
                      const Eigen::Matrix3d& Coarse_Rcl,
                      const Eigen::Vector3d& Coarse_tcl,
                      const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& filtered_clouds,
                      Eigen::Matrix3d& Optimized_Rcl,
                      Eigen::Vector3d& Optimized_tcl) {
    
    // Convert plane parameters to double precision.
    std::vector<Eigen::Vector4d> planes_double;
    for(const auto& p : camera_planes) planes_double.emplace_back(p.cast<double>());

    // Store data for later joint optimization.
    FrameData frame_data;
    frame_data.camera_planes = planes_double;
    frame_data.clouds = filtered_clouds;
    frame_data.initial_Rcl = Coarse_Rcl;
    frame_data.initial_tcl = Coarse_tcl;
    frames_data_.push_back(frame_data);

    // Run single-frame optimization.
    bool ok = OptimizeSingleFrame(planes_double, filtered_clouds, Coarse_Rcl, Coarse_tcl, Optimized_Rcl, Optimized_tcl);
    if (ok) {
        lidar_planes_ = TransformPlanesToLidarSingleFrame(planes_double, Optimized_Rcl, Optimized_tcl);
    }
    return ok;
}

// Single-frame optimization.
bool Coarse2Fine::OptimizeSingleFrame(const std::vector<Eigen::Vector4d>& planes_cam,
                                      const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& clouds,
                                      const Eigen::Matrix3d& Rcl_init,
                                      const Eigen::Vector3d& tcl_init,
                                      Eigen::Matrix3d& Rcl_opt,
                                      Eigen::Vector3d& tcl_opt) {
    double r_vec[3], t_vec[3];
    
    Eigen::AngleAxisd aa(Rcl_init);
    Eigen::Vector3d rv = aa.angle() * aa.axis();
    r_vec[0] = rv(0); r_vec[1] = rv(1); r_vec[2] = rv(2);
    t_vec[0] = tcl_init(0); t_vec[1] = tcl_init(1); t_vec[2] = tcl_init(2);

    ceres::Problem problem;
    
    for (size_t i = 0; i < 3; ++i) {
        for (const auto& pt : clouds[i]->points) {
            // Residual: point-to-plane distance; parameters: rotation and translation.
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<PlaneCostFunctor, 1, 3, 3>(
                    new PlaneCostFunctor(Eigen::Vector3d(pt.x, pt.y, pt.z), planes_cam[i]));
            
            problem.AddResidualBlock(cost_function, nullptr, r_vec, t_vec);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector3d r_vec_opt(r_vec[0], r_vec[1], r_vec[2]);
    Rcl_opt = ExpSO3(r_vec_opt);
    tcl_opt = Eigen::Vector3d(t_vec[0], t_vec[1], t_vec[2]);

    return summary.termination_type != ceres::FAILURE;
}

std::array<Eigen::Vector4f, 3> Coarse2Fine::TransformPlanesToLidarSingleFrame(
    const std::vector<Eigen::Vector4d>& planes_cam,
    const Eigen::Matrix3d& Rcl_opt,
    const Eigen::Vector3d& tcl_opt) {
    std::array<Eigen::Vector4f, 3> planes_lidar;
    for (size_t i = 0; i < planes_lidar.size(); ++i) {
        const auto& plane_cam = planes_cam[i];
        Eigen::Vector3d n_cam = plane_cam.head<3>();
        Eigen::Vector3d n_lidar = Rcl_opt.transpose() * n_cam;
        double d_lidar = n_cam.dot(tcl_opt) + plane_cam(3);
        planes_lidar[i] = Eigen::Vector4f(static_cast<float>(n_lidar(0)),
                                          static_cast<float>(n_lidar(1)),
                                          static_cast<float>(n_lidar(2)),
                                          static_cast<float>(d_lidar));
    }
    return planes_lidar;
}

const std::array<Eigen::Vector4f, 3>& Coarse2Fine::GetLidarPlanes() const {
    return lidar_planes_;
}

// Multi-frame joint optimization.
bool Coarse2Fine::JointOptimize(Eigen::Matrix3d& Final_Rcl, Eigen::Vector3d& Final_tcl) {
    if (frames_data_.empty()) {
        std::cerr << "Coarse2Fine::JointOptimize: No frames data!" << std::endl;
        return false;
    }
    // Use the refined parameters from the first frame as the initial estimate.
    Eigen::Matrix3d R_init = frames_data_[0].initial_Rcl;
    Eigen::Vector3d t_init = frames_data_[0].initial_tcl;

    double joint_r_vec[3], joint_t_vec[3];
    Eigen::AngleAxisd aa(R_init);
    Eigen::Vector3d rv = aa.angle() * aa.axis();
    joint_r_vec[0] = rv(0); joint_r_vec[1] = rv(1); joint_r_vec[2] = rv(2);
    joint_t_vec[0] = t_init(0); joint_t_vec[1] = t_init(1); joint_t_vec[2] = t_init(2);

    ceres::Problem joint_problem;
    // Joint optimization over point clouds from all frames.
    for (const auto& frame : frames_data_) {
        for (size_t i = 0; i < 3; ++i) {
            for (const auto& pt : frame.clouds[i]->points) {
                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<PlaneCostFunctor, 1, 3, 3>(
                        new PlaneCostFunctor(Eigen::Vector3d(pt.x, pt.y, pt.z), frame.camera_planes[i]));
                joint_problem.AddResidualBlock(cost_function, nullptr, joint_r_vec, joint_t_vec);
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 300;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    std::cout << "Solving joint optimization problem with " << frames_data_.size() << " frames..." << std::endl;
    ceres::Solve(options, &joint_problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    Eigen::Vector3d joint_rv(joint_r_vec[0], joint_r_vec[1], joint_r_vec[2]);
    Final_Rcl = ExpSO3(joint_rv);
    Final_tcl = Eigen::Vector3d(joint_t_vec[0], joint_t_vec[1], joint_t_vec[2]);

    return summary.termination_type != ceres::FAILURE;
}

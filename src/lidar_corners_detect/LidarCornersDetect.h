#ifndef _COARSE_CAL_H_
#define _COARSE_CAL_H_

#include <vector>
#include <array>
#include <utility>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "PointCloudUtil.h"

class LidarCornersDetect {
private:

    // Calibration board input parameters
    int rows_;
    int cols_;
    double square_len_;
    std::pair<double, double> origin_corner_uv_;
    std::array<std::pair<int, int>, 3> lines_plane_pairs_;

    std::array<pcl::PointCloud<pcl::PointXYZI>::Ptr, 3> corners_three_boards;
    std::vector<std::pair<LineEquation, LineEquation>> line_equations_;


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
    // Transform checkerboard points from (u, v) coordinates to the lidar frame.
    void transformCheckerBoardToLidarFrame(const PlaneBasis& basis, const std::vector<std::vector<Eigen::Vector2d>>& corners_uv, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

    void calculateCornersSingleBoard(const PlaneBasis& basis, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

public:
    LidarCornersDetect(int rows, int cols, double square_len, const std::pair<double, double>& origin_corner_uv, const std::array<std::pair<int, int>, 3>& lines_plane_pairs);

    void add(const std::array<Eigen::Vector4f, 3>& planes);

    const std::array<pcl::PointCloud<pcl::PointXYZI>::Ptr, 3>& GetCornersThreeBoard() const;

    const std::vector<std::pair<LineEquation, LineEquation>>& GetLineEquations() const;

    pcl::PointCloud<pcl::PointXYZI>::Ptr GetMergedCornersCloud() const;
};

#endif


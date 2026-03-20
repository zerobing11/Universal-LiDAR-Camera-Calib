#include "LidarCornersDetect.h"
#include <Eigen/Dense>
//计算三平面交线交点
void LidarCornersDetect::ComputePlaneIntersectionInfo(const std::array<Eigen::Vector4f, 3>& planes,
                                       const std::array<std::pair<int, int>, 3>& lines_plane_pairs,
                                       Eigen::Vector3d& intersection_point) {
    line_equations_.clear();
    line_equations_.reserve(3);
    
    auto compute_line = [](const Eigen::Vector4f& plane1,
                           const Eigen::Vector4f& plane2,
                           const Eigen::Vector3d& orient_normal,
                           LineEquation& line) -> void {
        Eigen::Vector3d n1 = plane1.head<3>().cast<double>();
        Eigen::Vector3d n2 = plane2.head<3>().cast<double>();
        Eigen::Vector3d dir = n1.cross(n2);
        double dir_norm_sq = dir.squaredNorm();
        if (dir_norm_sq < 1e-12) return;
        double d1 = plane1[3];
        double d2 = plane2[3];
        Eigen::Vector3d temp = (d2 * n1 - d1 * n2);
        Eigen::Vector3d point = temp.cross(dir) / dir_norm_sq;
        Eigen::Vector3d dir_normalized = dir.normalized();
        Eigen::Vector3d orient = orient_normal.normalized();
        if (dir_normalized.dot(orient) < 0.0) dir_normalized = -dir_normalized;
        line.direction = dir_normalized;
        line.point = point;
        return;
    };

    for(int i = 0; i < 3; i++)
    {
        int plane_a = lines_plane_pairs[i].first;   // line_u 对应的另一个平面
        int plane_b = lines_plane_pairs[i].second;  // line_v 对应的另一个平面
        
        LineEquation line_u, line_v;

        // 计算line_u
        int third_plane_u = 3 - i - plane_a;
        Eigen::Vector3d n_third_u = planes[third_plane_u].head<3>().cast<double>();
        compute_line(planes[i], planes[plane_a], n_third_u, line_u);

        // 计算line_v
        int third_plane_v = 3 - i - plane_b;
        Eigen::Vector3d n_third_v = planes[third_plane_v].head<3>().cast<double>();
        compute_line(planes[i], planes[plane_b], n_third_v, line_v);
        
        line_equations_.push_back({line_u, line_v});
    }

    Eigen::Matrix3d normals;
    Eigen::Vector3d rhs;
    for (int i = 0; i < 3; i++) {
        normals.row(i) = planes[i].head<3>().cast<double>();
        rhs[i] = -static_cast<double>(planes[i][3]);
    }
    intersection_point = normals.colPivHouseholderQr().solve(rhs);
}

void LidarCornersDetect::buildPlaneBasis(const Eigen::Vector4f& plane,
                     const LineEquation& line_i,
                     const LineEquation& line_j,
                     const Eigen::Vector3d& intersection_point,
                     PlaneBasis& basis) {
    Eigen::Vector3f n(plane[0], plane[1], plane[2]);
    Eigen::Vector3f x = line_i.direction.cast<float>();
    x.normalize();
    Eigen::Vector3f y = x.cross(n);
    y.normalize();
    Eigen::Vector3f jdir = line_j.direction.cast<float>();
    if (y.dot(jdir) < 0.0f) y = -y;
    basis.origin = intersection_point.cast<float>();
    basis.u = x;
    basis.v = y;
    basis.plane = plane;
    return;
}
//uv系转雷达系
void LidarCornersDetect::transformCheckerBoardToLidarFrame(const PlaneBasis& basis, const std::vector<std::vector<Eigen::Vector2d>>& corners_uv, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    if (!cloud) {
        cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    }
    cloud->clear();
    cloud->points.resize(rows_ * cols_);
    cloud->width = cols_;
    cloud->height = rows_;
    cloud->is_dense = true;
    //先行后列
    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            const auto& uv = corners_uv[r][c];
            Eigen::Vector3f p = basis.origin +
                                static_cast<float>(uv.x()) * basis.u +
                                static_cast<float>(uv.y()) * basis.v;
            pcl::PointXYZI pt; pt.x = p.x(); pt.y = p.y(); pt.z = p.z(); pt.intensity = 0.f;
            const size_t idx = r * cols_ + c;
            cloud->points[idx] = pt;
        }
    }
}
//计算雷达系下角点坐标
void LidarCornersDetect::calculateCornersSingleBoard(const PlaneBasis& basis, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    std::vector<std::vector<Eigen::Vector2d>> corners_uv(rows_, std::vector<Eigen::Vector2d>(cols_));
    double u0 = origin_corner_uv_.first;
    double v0 = origin_corner_uv_.second;

    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            // 按照先行后列的顺序
            double u = u0 + c * square_len_;
            double v = v0 + r * square_len_;
            corners_uv[r][c] = Eigen::Vector2d(u, v);
        }
    }
    transformCheckerBoardToLidarFrame(basis, corners_uv, cloud);
}

LidarCornersDetect::LidarCornersDetect(int rows, int cols, double square_len, const std::pair<double, double>& origin_corner_uv, const std::array<std::pair<int, int>, 3>& lines_plane_pairs)
    : rows_(rows), cols_(cols), square_len_(square_len), origin_corner_uv_(origin_corner_uv), lines_plane_pairs_(lines_plane_pairs) {
    for (auto& board : corners_three_boards) {
        board.reset(new pcl::PointCloud<pcl::PointXYZI>());
    }
}

void LidarCornersDetect::add(const std::array<Eigen::Vector4f, 3>& planes) {
    
    for (auto& board : corners_three_boards) {
        if (!board) {
            board.reset(new pcl::PointCloud<pcl::PointXYZI>());
        }
        board->clear();
    }
    //计算交线、交点
    Eigen::Vector3d intersection_point;
    ComputePlaneIntersectionInfo(planes,
        lines_plane_pairs_,
        intersection_point//out
        );

    for (int i = 0; i < 3; ++i) {
        PlaneBasis basis;
        //构建平面基准坐标系
        buildPlaneBasis(planes[i], 
                        line_equations_[i].first,
                        line_equations_[i].second,
                        intersection_point,
                        basis//out
                        );
        //根据起始角点uv坐标和棋盘格尺寸计算所有角点uv，并转换为lidar坐标系
        pcl::PointCloud<pcl::PointXYZI>::Ptr board_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        calculateCornersSingleBoard(basis, board_cloud);
        corners_three_boards[i] = board_cloud;
    }
    return;
}

const std::array<pcl::PointCloud<pcl::PointXYZI>::Ptr, 3>& LidarCornersDetect::GetCornersThreeBoard() const {
    return corners_three_boards;
}

const std::vector<std::pair<LineEquation, LineEquation>>& LidarCornersDetect::GetLineEquations() const {
    return line_equations_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr LidarCornersDetect::GetMergedCornersCloud() const {
    pcl::PointCloud<pcl::PointXYZI>::Ptr merged(new pcl::PointCloud<pcl::PointXYZI>());
    for (const auto& board : corners_three_boards) {
        if (board && !board->empty()) {
            *merged += *board;
        }
    }
    return merged;
}


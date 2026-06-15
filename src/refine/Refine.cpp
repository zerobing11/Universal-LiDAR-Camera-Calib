#include "Refine.h"
#include <Eigen/Dense>
//计算三平面交线交点
void Refine::ComputePlaneIntersectionInfo(const std::array<Eigen::Vector4f, 3>& planes,
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

void Refine::buildPlaneBasis(const Eigen::Vector4f& plane,
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
void Refine::transformCheckerBoardToLidarFrame(const PlaneBasis& basis, const std::vector<std::vector<Eigen::Vector2d>>& corners_uv, std::vector<std::vector<pcl::PointXYZI>>& corners) {
    int rows = corners_uv.size();
    int cols = corners_uv[0].size();
    if(corners.size() != rows || corners[0].size() != cols) {
        corners.assign(rows, std::vector<pcl::PointXYZI>(cols));
    }

    //先行后列
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const auto& uv = corners_uv[r][c];
            Eigen::Vector3f p = basis.origin +
                                static_cast<float>(uv.x()) * basis.u +
                                static_cast<float>(uv.y()) * basis.v;
            pcl::PointXYZI pt; pt.x = p.x(); pt.y = p.y(); pt.z = p.z(); pt.intensity = 0.f;
            corners[r][c] = pt;
        }
    }
}
bool Refine::splitCorners(const pcl::PointCloud<pcl::PointXYZI>::Ptr& corners_all,
                          std::vector<std::vector<pcl::PointXYZI>>& corners_split) {
    corners_split.clear();
    if (!corners_all) {
        return false;
    }
    int points_per_board = rows_ * cols_;
    if (static_cast<int>(corners_all->size()) != 3 * points_per_board) {
        return false;
    }
    corners_split.resize(3);
    for (int i = 0; i < 3; ++i) {
        auto begin_it = corners_all->points.begin() + i * points_per_board;
        auto end_it = corners_all->points.begin() + (i + 1) * points_per_board;
        corners_split[i].assign(begin_it, end_it);
    }
    return true;
}

//计算雷达系下角点坐标
void Refine::alignCorners(PlaneBasis& basis,
                          const std::vector<pcl::PointXYZI>& corners_cam2lidar,
                          std::vector<std::vector<pcl::PointXYZI>>& corners)
{
    std::vector<std::vector<Eigen::Vector2d>> corners_uv(rows_, std::vector<Eigen::Vector2d>(cols_));
    //计算uv系下标准坐标
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
    Eigen::Vector3d origin = basis.origin.cast<double>();
    Eigen::Vector3d u_axis = basis.u.cast<double>();
    Eigen::Vector3d v_axis = basis.v.cast<double>();

    std::vector<Eigen::Vector2d> src_pts, tgt_pts;
    int count = rows_ * cols_;
    //初始化两对待对齐点对
    src_pts.reserve(count);
    tgt_pts.reserve(count);
    Eigen::Vector2d src_center = Eigen::Vector2d::Zero();
    Eigen::Vector2d tgt_center = Eigen::Vector2d::Zero();
    for (int r = 0; r < rows_; ++r)
    {
        for (int c = 0; c < cols_; ++c)
        {
            // Src:corners_uv
            Eigen::Vector2d p_src_2d = corners_uv[r][c];

            // Tgt:corners_cam2lidar投影到uv系
            int idx = r * cols_ + c;
            if(idx >= corners_cam2lidar.size()) continue;

            const auto& p_tgt_3d = corners_cam2lidar[idx];
            Eigen::Vector3d v_tgt(p_tgt_3d.x, p_tgt_3d.y, p_tgt_3d.z);
            Eigen::Vector3d diff_tgt = v_tgt - origin;
            Eigen::Vector2d p_tgt_2d(diff_tgt.dot(u_axis), diff_tgt.dot(v_axis));

            src_pts.push_back(p_src_2d);
            tgt_pts.push_back(p_tgt_2d);
            src_center += p_src_2d;
            tgt_center += p_tgt_2d;
        }
    }
    //对齐
    count = src_pts.size();
    src_center /= static_cast<double>(count);
    tgt_center /= static_cast<double>(count);
    Eigen::Matrix2d W = Eigen::Matrix2d::Zero();
    for (int i = 0; i < count; ++i)
    {
        W += (tgt_pts[i] - tgt_center) * (src_pts[i] - src_center).transpose();
    }
    //SVD分解求2D旋转和平移
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d U = svd.matrixU();
    Eigen::Matrix2d V = svd.matrixV();
    Eigen::Matrix2d R_2d = U * V.transpose();
    if (R_2d.determinant() < 0)
    {
        U.col(1) *= -1;
        R_2d = U * V.transpose();
    }
    Eigen::Vector2d t_2d = tgt_center - R_2d * src_center;
    // 变换并更新 Basis
    Eigen::Vector3d new_origin = origin + t_2d.x() * u_axis + t_2d.y() * v_axis;
    Eigen::Vector3d new_u_axis = R_2d(0, 0) * u_axis + R_2d(1, 0) * v_axis;
    Eigen::Vector3d new_v_axis = R_2d(0, 1) * u_axis + R_2d(1, 1) * v_axis;
    basis.origin = new_origin.cast<float>();
    basis.u = new_u_axis.cast<float>();
    basis.v = new_v_axis.cast<float>();
    //利用更新完的basis和corners_uv，将corners_uv变换到雷达系下
    transformCheckerBoardToLidarFrame(basis, corners_uv, corners);
}

Refine::Refine(int rows, int cols, double square_len, const std::pair<double, double>& origin_corner_uv, const std::array<std::pair<int, int>, 3>& lines_plane_pairs)
    : rows_(rows), cols_(cols), square_len_(square_len), origin_corner_uv_(origin_corner_uv), lines_plane_pairs_(lines_plane_pairs) {
    for (auto& board : perfect_checkerboards) {
        board.resize(rows_, cols_);
    }
}

void Refine::growCorners(PerfectCheckerBoard& board) {
    Eigen::Vector3d u_dir = board.basis.u.cast<double>();
    Eigen::Vector3d v_dir = board.basis.v.cast<double>();
    // board.corners_growth (rows_ + 2) x (cols_ + 2)
    int rows_new = rows_ + 2;
    int cols_new = cols_ + 2;
    for (int r = 0; r < rows_new; ++r) {
        for (int c = 0; c < cols_new; ++c) {
             //内部点直接拷贝
             bool is_inner_row = (r >= 1 && r <= rows_new - 2);
             bool is_inner_col = (c >= 1 && c <= cols_new - 2);
             if (is_inner_row && is_inner_col) {
                 board.corners_growth[r][c] = board.corners[r - 1][c - 1];
             }
            // 边界扩展
            else {
                 int rr = std::max(1, std::min(r, rows_new - 2));
                 int cc = std::max(1, std::min(c, cols_new - 2));
                 const auto& ref_corner = board.corners[rr - 1][cc - 1];
                 int dr = r - rr;
                 int dc = c - cc;
                 Eigen::Vector3d offset = static_cast<double>(dc) * u_dir * square_len_ +
                                          static_cast<double>(dr) * v_dir * square_len_;
                 pcl::PointXYZI pt;
                 pt.x = ref_corner.x + static_cast<float>(offset.x());
                 pt.y = ref_corner.y + static_cast<float>(offset.y());
                 pt.z = ref_corner.z + static_cast<float>(offset.z());
                 pt.intensity = ref_corner.intensity;
                 board.corners_growth[r][c] = pt;
             }
        }
    }
}

void Refine::computeCellCentroids(PerfectCheckerBoard& board) {
    // board.cell_centroids_growth(rows_+1, cols_+1)
    for (int r = 0; r < rows_ + 1; ++r) {
        for (int c = 0; c < cols_ + 1; ++c) {
             const auto& p_tl = board.corners_growth[r][c];
             const auto& p_tr = board.corners_growth[r][c+1];
             const auto& p_bl = board.corners_growth[r+1][c];
             const auto& p_br = board.corners_growth[r+1][c+1];

             pcl::PointXYZI centroid;
             centroid.x = (p_tl.x + p_tr.x + p_bl.x + p_br.x) / 4.0f;
             centroid.y = (p_tl.y + p_tr.y + p_bl.y + p_br.y) / 4.0f;
             centroid.z = (p_tl.z + p_tr.z + p_bl.z + p_br.z) / 4.0f;
             centroid.intensity = 0.f;
             
             board.cell_centroids_growth[r][c] = centroid;
        }
    }
}

void Refine::computeCellColors(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, PerfectCheckerBoard& board) {
    if (!cloud || cloud->empty()) return;

    pcl::KdTreeFLANN<pcl::PointXYZINormal> kdtree;
    kdtree.setInputCloud(cloud);

    std::vector<int> indices;
    std::vector<float> sqr_dists;
    float radius = static_cast<float>(square_len_ / 3.0);

    int rows_growth = board.cell_centroids_growth.size();
    if (rows_growth == 0) return;
    int cols_growth = board.cell_centroids_growth[0].size();

    for (int r = 0; r < rows_growth; ++r) {
        for (int c = 0; c < cols_growth; ++c) {
            auto& centroid = board.cell_centroids_growth[r][c];
            pcl::PointXYZINormal searchPoint;
            searchPoint.x = centroid.x;
            searchPoint.y = centroid.y;
            searchPoint.z = centroid.z;

            indices.clear();
            sqr_dists.clear();
            //找附近radius点云上色
            int found = kdtree.radiusSearch(searchPoint, radius, indices, sqr_dists);
            if (found > 0) {
                double sum = 0.0;
                for (int idx : indices) {
                    sum += cloud->points[idx].intensity;
                }
                float avg_intensity = static_cast<float>(sum / static_cast<double>(found));
                centroid.intensity = avg_intensity;
                board.cell_centroids_color[r][c] = static_cast<double>(avg_intensity);
            }
        }
    }
}
// 计算入射角写入normal_z，并根据入射角和平均发散角K_deg计算光斑半径写入curvature
void Refine::computeAngleRadiusAndFilter(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud,
                                const Eigen::Vector4f& plane,
                                double K_deg,
                                double angle_threshold) {
    Eigen::Vector3d n(static_cast<double>(plane[0]),
                      static_cast<double>(plane[1]),
                      static_cast<double>(plane[2]));
    const double n_norm = n.norm();
    n /= n_norm;
    const double K_rad = K_deg * M_PI / 180.0;
    for (auto& pt : cloud->points) {
        Eigen::Vector3d v(static_cast<double>(pt.x),
                          static_cast<double>(pt.y),
                          static_cast<double>(pt.z));
        const double range = v.norm();
        v /= range;
        // cos(theta)=|n·v|  n为平面法向，v为点云方向
        double cos_v = std::abs(n.dot(v));
        cos_v = std::max(-1.0, std::min(1.0, cos_v));
        const double ang_deg = std::acos(cos_v) * 180.0 / M_PI;

        double curvature = 0.0;
        // radius = range * K_rad / cos(theta)  range为点云距离，K_rad为平均发散角
        curvature = range * K_rad / cos_v;
        pt.normal_x = 0.0f;
        pt.normal_y = 0.0f;
        pt.normal_z = static_cast<float>(ang_deg);
        pt.curvature = static_cast<float>(curvature);
    }

    // 过滤normal_z 大于angle_threshold 的点
    auto new_end = std::remove_if(cloud->points.begin(), cloud->points.end(),
        [angle_threshold](const pcl::PointXYZINormal& pt) {
            return pt.normal_z > angle_threshold;
        });
    cloud->points.erase(new_end, cloud->points.end());

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
}

void Refine::add(const std::array<Eigen::Vector4f, 3>& planes,
                 const pcl::PointCloud<pcl::PointXYZI>::Ptr& corners_cam2lidar,
                 std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& clouds) {
    //定长
    for (auto& board : perfect_checkerboards) {
        board.resize(rows_, cols_);
        board.rect = square_len_; //设置棋盘格边长
    }
    //计算交线、交点
    Eigen::Vector3d intersection_point;
    ComputePlaneIntersectionInfo(planes,
        lines_plane_pairs_,
        intersection_point//out
        );
    //角点分离
    std::vector<std::vector<pcl::PointXYZI>> corners_split;
    if (!splitCorners(corners_cam2lidar, corners_split)) {
        cout<<"Split corners failed"<<endl;
        return;
    }
    for (int i = 0; i < 3; ++i) {
        //构建单面基准坐标系
        buildPlaneBasis(planes[i],
                        line_equations_[i].first,
                        line_equations_[i].second,
                        intersection_point,
                        perfect_checkerboards[i].basis//out
                        );
        //将标准棋盘角点与corners_cam2lidar进行对齐
        alignCorners(perfect_checkerboards[i].basis, corners_split[i], perfect_checkerboards[i].corners);
        //角点corners扩成corners_growth
        growCorners(perfect_checkerboards[i]);
        //cell_centroids_growth质心求解
        computeCellCentroids(perfect_checkerboards[i]);
        //质心上色
        computeCellColors(clouds[i], perfect_checkerboards[i]);
        //入射角过滤与计算光斑半径
        computeAngleRadiusAndFilter(clouds[i],planes[i],0.15,70);
        //更新原点坐标！！！！！！
        perfect_checkerboards[i].basis.origin = {perfect_checkerboards[i].corners_growth[0][0].x,perfect_checkerboards[i].corners_growth[0][0].y, perfect_checkerboards[i].corners_growth[0][0].z};

    }
    //进行优化
    Eigen::Matrix3d R_init = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_init = Eigen::Vector3d::Zero();
    lm_result_ = OptimizePlanePose(clouds, perfect_checkerboards, R_init, t_init);
    std::cout << "迭代次数: " << lm_result_.iterations
    << " 代价: " << lm_result_.final_cost << std::endl;
    // std::cout << "delta_R: " <<endl<< lm_result_.R << std::endl;
    // std::cout << "delta_t: " <<endl<< lm_result_.t.transpose() << std::endl;
    std::cout << "light-spot radius factor k: " << lm_result_.k << std::endl;
    //变换平面方程和角点质心点云
    computeInverseTransform(planes);

    return;
}

const std::array<PerfectCheckerBoard, 3>& Refine::GetPerfectCheckerboards() const {
    return perfect_checkerboards;
}

const std::vector<std::pair<LineEquation, LineEquation>>& Refine::GetLineEquations() const {
    return line_equations_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr Refine::GetCornersCloud() const {
    pcl::PointCloud<pcl::PointXYZI>::Ptr corners(new pcl::PointCloud<pcl::PointXYZI>());
    for (const auto& board : perfect_checkerboards) {
        for (const auto& row : board.corners) {
            for (const auto& pt : row) {
                corners->push_back(pt);
            }
        }
    }
    corners->width = corners->points.size();
    corners->height = 1;
    corners->is_dense = true;
    return corners;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr Refine::GetCentroidsCloud() const {
    pcl::PointCloud<pcl::PointXYZI>::Ptr centroids(new pcl::PointCloud<pcl::PointXYZI>());
    for (const auto& board : perfect_checkerboards) {
        for (const auto& row : board.cell_centroids_growth) {
            for (const auto& pt : row) {
                centroids->push_back(pt);
            }
        }
    }
    centroids->width = centroids->points.size();
    centroids->height = 1;
    centroids->is_dense = true;
    return centroids;
}

const LMResult& Refine::GetLMResult() const {
    return lm_result_;
}

LMResult Refine::OptimizePlanePose(const std::array<pcl::PointCloud<pcl::PointXYZINormal>::Ptr, 3>& clouds,
                                   const std::array<PerfectCheckerBoard, 3>& boards,
                                   const Eigen::Matrix3d& R0,
                                   const Eigen::Vector3d& t0) {
    LMResult result;

    // 初始化优化变量：旋转向量 (3维) 和 平移向量 (3维)
    double r_vec[3], t_vec[3];
    double k_val = 3;
    Eigen::AngleAxisd aa(R0);
    Eigen::Vector3d rv = aa.angle() * aa.axis();
    r_vec[0] = rv(0); r_vec[1] = rv(1); r_vec[2] = rv(2);
    t_vec[0] = t0(0); t_vec[1] = t0(1); t_vec[2] = t0(2);

    ceres::Problem problem;
    // 遍历三个平面点云，将所有残差添加到同一个优化问题中
    for (size_t i = 0; i < 3; ++i) {
        const auto& cloud = clouds[i];
        const auto& board = boards[i];

        for (const auto& pt : cloud->points) {
            // 读取观测信息
            Eigen::Vector3d p(pt.x, pt.y, pt.z);
            double radius = static_cast<double>(pt.curvature);
            double intensity_world = static_cast<double>(pt.intensity);

            // 使用 CheckerboardCostFunctor
            // <1, 3, 3, 1>: 残差维度 1, 旋转参数 3, 平移参数 3, k参数 1
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<CheckerboardCostFunctor, 1, 3, 3, 1>(
                    new CheckerboardCostFunctor(p, intensity_world, radius, &board));

            problem.AddResidualBlock(cost_function, nullptr, r_vec, t_vec, &k_val);
        }
    }

    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 获取优化结果
    Eigen::Vector3d r_opt(r_vec[0], r_vec[1], r_vec[2]);
    Eigen::Vector3d t_opt(t_vec[0], t_vec[1], t_vec[2]);

    result.R = ExpSO3(r_opt);
    result.t = t_opt;
    result.k = k_val;
    result.final_cost = summary.final_cost;
    result.iterations = summary.iterations.size();
    return result;
}

void Refine::computeInverseTransform(const std::array<Eigen::Vector4f, 3>& planes) {
    // 计算逆变换矩阵
    Eigen::Matrix3d R_inv = lm_result_.R.transpose();
    Eigen::Vector3d t_inv = -R_inv * lm_result_.t;

    // 逆变换平面方程
    for (int pid = 0; pid < 3; ++pid) {
        // 原平面参数
        Eigen::Vector3d n_orig = planes[pid].head<3>().cast<double>();
        double d_orig = static_cast<double>(planes[pid][3]);

        // 变换平面: n' = R_inv * n, d' = d - n' * t_inv
        Eigen::Vector3d n_new = R_inv * n_orig;
        double d_new = d_orig - n_new.dot(t_inv);

        planes_refine_[pid] << static_cast<float>(n_new.x()),
                            static_cast<float>(n_new.y()), 
                            static_cast<float>(n_new.z()), 
                            static_cast<float>(d_new);
    }

    // 角点点云逆变换
    pcl::PointCloud<pcl::PointXYZI>::Ptr corners_cloud = GetCornersCloud();
    corners_cloud_refine_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    if (corners_cloud && !corners_cloud->empty()) {
        pcl::copyPointCloud(*corners_cloud, *corners_cloud_refine_);
        for (auto& pt : corners_cloud_refine_->points) {
            Eigen::Vector3d p(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_new = R_inv * p + t_inv;
            pt.x = static_cast<float>(p_new.x());
            pt.y = static_cast<float>(p_new.y());
            pt.z = static_cast<float>(p_new.z());
        }
    }

    // 质心点云逆变换
    pcl::PointCloud<pcl::PointXYZI>::Ptr centroids_cloud = GetCentroidsCloud();
    centroids_cloud_refine_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    if (centroids_cloud && !centroids_cloud->empty()) {
        pcl::copyPointCloud(*centroids_cloud, *centroids_cloud_refine_);
        for (auto& pt : centroids_cloud_refine_->points) {
            Eigen::Vector3d p(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_new = R_inv * p + t_inv;
            pt.x = static_cast<float>(p_new.x());
            pt.y = static_cast<float>(p_new.y());
            pt.z = static_cast<float>(p_new.z());
        }
    }
}

const std::array<Eigen::Vector4f, 3>& Refine::GetPlanesRefine() const {
    return planes_refine_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr Refine::GetCornersCloudRefine() const {
    return corners_cloud_refine_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr Refine::GetCentroidsCloudRefine() const {
    return centroids_cloud_refine_;
}


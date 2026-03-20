#include "Camera.h"

namespace CAMERA
{

Camera::Camera(int corners_col_,int corners_row_,string path,double corner_thresh,double chess_thresh,
               float square_size, const cv::Mat& intrinsic_matrix, const cv::Mat& distortion_coeffs,const std::array<std::pair<int, int>, 3> line_plane_pairs)
{
    path_root=path;
    numofcorner=corners_col_*corners_row_;

    // 设置自定义阈值
    corner_detect_threshold = corner_thresh;
    chessboard_threshold = chess_thresh;
    initialization=false;
    img_indx=-1;
    corners_col=corners_col_;
    corners_row=corners_row_;
    distParameter = Mat(1,5,CV_64FC1,Scalar::all(0));

    // 保存用于PnP排序的参数
    square_size_ = square_size;
    intrinsic_for_sort_ = intrinsic_matrix.clone();
    distortion_for_sort_ = distortion_coeffs.clone();
    line_plane_pairs_ = line_plane_pairs;
    // 初始化单个棋盘格系下的3D角点坐标
    board_3d_points_.clear();
    for (int row = 0; row < corners_row; row++)
    {
        for (int col = 0; col < corners_col; col++)
        {
            cv::Point3f pt;
            pt.x = col * square_size_;
            pt.y = row * square_size_;
            pt.z = 0;
            board_3d_points_.push_back(pt);
        }
    }
    cout << "init 3D checker: " << board_3d_points_.size() << " points, square_len: " << square_size_ << "m" << endl;
}

void Camera::init_img()
    {
    initialization=true;
    image_size.width = img.cols;
    image_size.height = img.rows;
    }

void Camera::DataClear()
    {
    chessboards.clear();
    corners_s.p.clear();
    corners_s.v1.clear();
    corners_s.v2.clear();
    corners_s.score.clear();
    
    // 清理 cur_boards 中的所有 vector 成员
    for(int i = 0; i < 3; i++)
    {
        cur_boards[i].corners.clear();
        cur_boards[i].orderd_corners.clear();
        cur_boards[i].corners_3d.clear();
        cur_boards[i].orderd_corners_3d.clear();
        cur_boards[i].origin_2d.clear();
        cur_boards[i].line_u = Eigen::Vector3f::Zero();
        cur_boards[i].line_v = Eigen::Vector3f::Zero();
        cur_boards[i].origin = Eigen::Vector3f::Zero();
        cur_boards[i].plane = Eigen::Vector4f::Zero();
    }
    }

bool Camera::Ensure_ValidFrame(std::vector<cv::Mat> chessboards)
{

    if((chessboards.size()==3)&&(chessboards[0].cols*chessboards[0].rows==numofcorner)&&
    (chessboards[1].cols*chessboards[1].rows==numofcorner)&&
    (chessboards[2].cols*chessboards[2].rows==numofcorner))
    {
        return true;
    }
    else
    {
        return false;
    }
}

// 可视化棋盘格，按排序后的顺序显示
void Camera::visualize_chessboards() {
    string corner_debug_dir = path_root + "/img_corner_test";
    string mkdir_cmd = "mkdir -p " + corner_debug_dir;
    system(mkdir_cmd.c_str());

    cv::Mat chessboard_vis_img = org_img.clone();
    vector<cv::Scalar> board_colors = {
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 0),
    };

    // 按排序后的平面顺序，用 cur_boards 中的角点顺序
    for(int board_idx = 0; board_idx < 3; board_idx++)
    {
        if(board_idx < 0 || board_idx >= 3 || board_idx >= chessboards.size()) continue;

        cv::Scalar color = board_colors[board_idx];
        const vector<cv::Point2f>& corners = cur_boards[board_idx].corners;
        
        // 绘制每个角点及其序号
        for(size_t i = 0; i < corners.size(); ++i)
        {
            const cv::Point2f& pt = corners[i];
            cv::circle(chessboard_vis_img, pt, 6.5, color, -1);
            cv::putText(chessboard_vis_img, std::to_string(i),
                       cv::Point2f(pt.x + 8, pt.y - 8),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2.5);
        }
        
        // // 在每个板子的2D角点起点上方标注 Plane_xxx
        // if(!corners.empty())
        // {
        //     cv::Point2f origin_pt = corners.front();
        //     string board_label = "Plane_" + std::to_string(board_idx + 1);
        //     cv::putText(chessboard_vis_img, board_label,
        //                cv::Point2f(origin_pt.x - 20, origin_pt.y - 20),
        //                cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        // }
    }

    string chessboard_save_path = corner_debug_dir + "/frame_" +
                                 std::to_string(img_indx) + "_chessboards_sorted.png";
    cv::imwrite(chessboard_save_path, chessboard_vis_img);
    cout << "saved sorted checker in: " << chessboard_save_path << endl;
}


// 可视化棋盘格，按排序后的顺序显示
void Camera::visualize_corners() {
    string corner_debug_dir = path_root + "/img_sorted_boards";
    string mkdir_cmd = "mkdir -p " + corner_debug_dir;
    system(mkdir_cmd.c_str());

    cv::Mat chessboard_vis_img = org_img.clone();
    vector<cv::Scalar> board_colors = {
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 0),
    };

    // 按排序后的平面顺序，用 cur_boards 中的角点顺序
    for(int board_idx = 0; board_idx < 3; board_idx++)
    {
        if(board_idx < 0 || board_idx >= 3 || board_idx >= chessboards.size()) continue;

        cv::Scalar color = board_colors[board_idx];
        const vector<cv::Point2f>& corners = cur_boards[board_idx].orderd_corners;

        // 绘制每个角点及其序号
        for(size_t i = 0; i < corners.size(); ++i)
        {
            const cv::Point2f& pt = corners[i];
            cv::circle(chessboard_vis_img, pt, 6.5, color, -1);
            cv::putText(chessboard_vis_img, std::to_string(i),
                       cv::Point2f(pt.x + 8, pt.y - 8),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2.5);
        }
    }

    string chessboard_save_path = corner_debug_dir + "/frame_" +
                                 std::to_string(img_indx) + "_chessboards_sorted.png";
    cv::imwrite(chessboard_save_path, chessboard_vis_img);
    cout << "saved sorted checker corners image in: " << chessboard_save_path << endl;
}

// 根据角点矩形区域生成遮罩并保存
void Camera::visualize_masks() {
    string mask_dir = path_root + "/img_mask";
    string mkdir_cmd = "mkdir -p " + mask_dir;
    system(mkdir_cmd.c_str());

    if (org_img.empty()) {
        return;
    }

    auto compute_mask_rect = [&](const vector<cv::Point2f>& corners)->cv::Rect {
        if (corners.empty()) {
            return cv::Rect();
        }
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();
        for (const auto& pt : corners) {
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }
        int x1 = static_cast<int>(std::floor(min_x)) - 10;
        int y1 = static_cast<int>(std::floor(min_y)) - 10;
        int x2 = static_cast<int>(std::ceil(max_x)) + 10;
        int y2 = static_cast<int>(std::ceil(max_y)) + 10;
        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        x2 = std::min(org_img.cols - 1, x2);
        y2 = std::min(org_img.rows - 1, y2);
        if (x2 <= x1 || y2 <= y1) {
            return cv::Rect();
        }
        return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
    };

    std::array<cv::Rect, 3> board_rects;
    for (int i = 0; i < 3; ++i) {
        board_rects[i] = compute_mask_rect(cur_boards[i].orderd_corners);
    }

    for (int keep_idx = 0; keep_idx < 3; ++keep_idx) {
        cv::Mat masked_img = org_img.clone();
        for (int board_idx = 0; board_idx < 3; ++board_idx) {
            if (board_idx == keep_idx) continue;
            if (board_rects[board_idx].area() <= 0) continue;
            cv::rectangle(masked_img, board_rects[board_idx], cv::Scalar(0, 0, 0), cv::FILLED);
        }
        string save_path = mask_dir + "/" + std::to_string(img_indx)
                         + "_" + std::to_string(keep_idx + 1) + ".png";
        cv::imwrite(save_path, masked_img);
    }
}


//提取棋盘格角点坐标,确保行列顺序正确
void Camera::extract_corners()
{
    for (int indx_bd = 0; indx_bd < 3; indx_bd++) {
        vector<cv::Point2f> base_pts;
        if (chessboards[indx_bd].rows == corners_row && chessboards[indx_bd].cols == corners_col) {
            for (int i = 0; i < chessboards[indx_bd].rows; i++) {
                for (int j = 0; j < chessboards[indx_bd].cols; j++)
                {
                    int d = chessboards[indx_bd].at<int>(i, j);
                    cv::Point2f point(corners_s.p[d].x, corners_s.p[d].y);
                    base_pts.push_back(point);
                }
            }
        }
        else {
            for (int col = 0; col < chessboards[indx_bd].cols; col++) {
                for (int row = 0; row < chessboards[indx_bd].rows; row++)
                {
                    int d = chessboards[indx_bd].at<int>(row, col);  // at(row, col)
                    cv::Point2f point(corners_s.p[d].x, corners_s.p[d].y);
                    base_pts.push_back(point);
                }
            }
        }

        // 因为棋盘格生长方向的不确定性，角点顺序可能是原始的、或者行倒序、或者列倒序等
        vector<vector<cv::Point2f>> candidates;
        vector<string> labels = {"Original", "Rotate180", "FlipX (RowRev)", "FlipY (ColRev)"};
        
        // 原本
        candidates.push_back(base_pts);

        //旋转 180
        vector<cv::Point2f> pts_180 = base_pts;
        std::reverse(pts_180.begin(), pts_180.end());
        candidates.push_back(pts_180);

        //翻转x轴镜像
        vector<cv::Point2f> pts_flip_x;
        for(int r = 0; r < corners_row; r++) {
            for(int c = corners_col - 1; c >= 0; c--) {
                pts_flip_x.push_back(base_pts[r * corners_col + c]);
            }
        }
        candidates.push_back(pts_flip_x);

        // 翻转y轴镜像
        vector<cv::Point2f> pts_flip_y;
        for(int r = corners_row - 1; r >= 0; r--) {
            for(int c = 0; c < corners_col; c++) {
                pts_flip_y.push_back(base_pts[r * corners_col + c]);
            }
        }
        candidates.push_back(pts_flip_y);

        // 对每种候选进行PnP求解，选择误差最小的
        int best_idx = 0;
        double min_error = std::numeric_limits<double>::max();
        
        for(int k=0; k<4; k++) {
            cv::Mat rvec, tvec;
            // 使用EPnP求解
            bool success = cv::solvePnP(board_3d_points_, candidates[k], intrinsic_for_sort_, distortion_for_sort_,
                                        rvec, tvec, false, cv::SOLVEPNP_EPNP);
            
            double err = 1e9;
            if(success) {
                vector<cv::Point2f> reproj_pts;
                cv::projectPoints(board_3d_points_, rvec, tvec, intrinsic_for_sort_, distortion_for_sort_, reproj_pts);
                double sum_err = 0;
                for(size_t i=0; i<reproj_pts.size(); i++) {
                    double dx = reproj_pts[i].x - candidates[k][i].x;
                    double dy = reproj_pts[i].y - candidates[k][i].y;
                    sum_err += std::sqrt(dx*dx + dy*dy);
                }
                err = sum_err / reproj_pts.size();
            }

            if(err < min_error) {
                min_error = err;
                best_idx = k;
            }
        }
        // cout << "  板 " << indx_bd << " 最佳方向: " << labels[best_idx] << " (Error: " << min_error << ")" << endl;
        cur_boards[indx_bd].corners = candidates[best_idx];
    }
}

// 计算三个标定板的交线方向向量
// line_plane_pairs[i] = {a, b} 表示：
// line_u: plane[i] 与 plane[a] 的交线
// line_v: plane[i] 与 plane[b] 的交线
void Camera::compute_line_model(const std::array<std::pair<int, int>, 3>& line_plane_pairs)
{
    for(int i = 0; i < 3; i++)
    {
        int plane_a = line_plane_pairs[i].first;   // line_u 对应的另一个平面
        int plane_b = line_plane_pairs[i].second;  // line_v 对应的另一个平面
        
        Eigen::Vector3f ni = cur_boards[i].plane.head<3>();
        Eigen::Vector3f na = cur_boards[plane_a].plane.head<3>();
        Eigen::Vector3f nb = cur_boards[plane_b].plane.head<3>();
        
        // 计算 plane[i] 与 plane[a] 的交线 -> line_u
        Eigen::Vector3f dir_u = ni.cross(na);
        float norm_u = dir_u.norm();
        if(norm_u > 1e-6)
        {
            dir_u /= norm_u;
            // 找第三个平面的索引用于调整方向
            int third_plane = 3 - i - plane_a;  // 因为 0+1+2=3
            Eigen::Vector3f n_third = cur_boards[third_plane].plane.head<3>();
            if(dir_u.dot(n_third) < 0.0f)
            {
                dir_u = -dir_u;
            }
        }
        
        // 计算 plane[i] 与 plane[b] 的交线 -> line_v
        Eigen::Vector3f dir_v = ni.cross(nb);
        float norm_v = dir_v.norm();
        if(norm_v > 1e-6)
        {
            dir_v /= norm_v;
            // 找第三个平面的索引用于调整方向
            int third_plane = 3 - i - plane_b;
            Eigen::Vector3f n_third = cur_boards[third_plane].plane.head<3>();
            if(dir_v.dot(n_third) < 0.0f)
            {
                dir_v = -dir_v;
            }
        }
        
        cur_boards[i].line_u = dir_u;  // plane[i] 与 plane[a] 的交线
        cur_boards[i].line_v = dir_v;  // plane[i] 与 plane[b] 的交线
    }
}

// 计算三块平面两两之间的法向夹角，并判断是否均大于80度
bool Camera::compute_plane_angle()
{
    auto clamp01 = [](float v){ return std::max(-1.0f, std::min(1.0f, v)); };
    auto angle_deg = [&](const Eigen::Vector3f& n1, const Eigen::Vector3f& n2)->float{
        float denom = n1.norm() * n2.norm();
        if(denom < 1e-9f) return 0.0f;
        float cos_theta = clamp01(n1.dot(n2) / denom);
        return static_cast<float>(std::acos(cos_theta) * 180.0 / CV_PI);
    };

    bool all_angles_gt_80 = true;
    for(int i = 0; i < 3; i++)
    {
        for(int j = i + 1; j < 3; j++)
        {
            Eigen::Vector3f ni = cur_boards[i].plane.head<3>();
            Eigen::Vector3f nj = cur_boards[j].plane.head<3>();
            float ang = angle_deg(ni, nj);
            cout << "angle_plane" << i+1<< j+1 << ": " << ang << " deg" << endl;
            if(ang <= 80.0f)
            {
                all_angles_gt_80 = false;
            }
        }
    }
    return all_angles_gt_80;
}

// 对三个标定板进行排序（相机系下3D坐标）
bool Camera::sort_boards()
{
    struct BoardFeature {
        int board_idx;          // 原始标定板索引
        float max_x;            // 角点在相机系下的最大x值
        float min_x;            // 角点在相机系下的最小x值
        float max_y;            // 角点在相机系下的最大y值
        vector<cv::Point3f> corners_in_cam;  // 角点在相机系下的坐标
    };
    vector<BoardFeature> board_features(3);

    // 对每个标定板使用PnP求解相机位姿，计算角点在相机系下的坐标
    for(int board_idx = 0; board_idx < 3; board_idx++)
    {
        board_features[board_idx].board_idx = board_idx;
        vector<cv::Point2f> image_pts = cur_boards[board_idx].corners;

        // 使用PnP求解相机在标定板系下的位姿
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(board_3d_points_, image_pts, intrinsic_for_sort_, distortion_for_sort_,
                                    rvec, tvec, false, cv::SOLVEPNP_EPNP);
        if(success)
        {
            success = cv::solvePnP(board_3d_points_, image_pts, intrinsic_for_sort_, distortion_for_sort_,
                                   rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
        }

        // 计算并打印重投影误差
        vector<cv::Point2f> reprojected_pts;
        cv::projectPoints(board_3d_points_, rvec, tvec, intrinsic_for_sort_, distortion_for_sort_, reprojected_pts);
        double reproj_error = 0.0;
        for(int i = 0; i < image_pts.size(); i++)
        {
            double dx = image_pts[i].x - reprojected_pts[i].x;
            double dy = image_pts[i].y - reprojected_pts[i].y;
            reproj_error += dx * dx + dy * dy;
        }
        if(!image_pts.empty())
        {
            reproj_error = std::sqrt(reproj_error / static_cast<double>(image_pts.size()));
        }
        cout << "unsorted checker image " << board_idx << " reproject-error-RMSE: " << reproj_error << " pixel" << endl;

        // 将标定板系下的角点转换到相机系下，存到板子结构体中并且计算角点在相机系下的最值
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        board_features[board_idx].max_x = -std::numeric_limits<float>::max();
        board_features[board_idx].min_x = std::numeric_limits<float>::max();
        board_features[board_idx].max_y = -std::numeric_limits<float>::max();
        for(int i = 0; i < board_3d_points_.size(); i++)
        {
            cv::Mat pt_board = (cv::Mat_<double>(3, 1) << board_3d_points_[i].x,
                                                          board_3d_points_[i].y,
                                                          board_3d_points_[i].z);
            cv::Mat pt_cam = R * pt_board + tvec;

            float x_cam = static_cast<float>(pt_cam.at<double>(0, 0));
            float y_cam = static_cast<float>(pt_cam.at<double>(1, 0));
            float z_cam = static_cast<float>(pt_cam.at<double>(2, 0));

            cur_boards[board_idx].corners_3d.push_back(cv::Point3f(x_cam, y_cam, z_cam));

            if(x_cam > board_features[board_idx].max_x)
                board_features[board_idx].max_x = x_cam;
            if(x_cam < board_features[board_idx].min_x)
                board_features[board_idx].min_x = x_cam;
            if(y_cam > board_features[board_idx].max_y)
                board_features[board_idx].max_y = y_cam;
        }
        
        // 计算相机系下的平面方程，标定板在自身坐标系的法向量就是001
        Eigen::Matrix3d Rcw;
        Rcw << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
               R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
               R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
        Eigen::Vector3d tcw;
        tcw << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
        
        Eigen::Vector3d normal_cam = Rcw.col(2);
        double norm_len = normal_cam.norm();
        if (norm_len > 1e-9) {
            normal_cam /= norm_len;
        }
        double d_plane = -normal_cam.dot(tcw);
        if (d_plane < 0) {        // 确保d>0
            normal_cam = -normal_cam;
            d_plane = -d_plane;
        }
        
        cur_boards[board_idx].plane = Eigen::Vector4f(
            static_cast<float>(normal_cam(0)),
            static_cast<float>(normal_cam(1)),
            static_cast<float>(normal_cam(2)),
            static_cast<float>(d_plane)
        );
    }

    // 平面排序：平面3为法向量与相机系-Y轴夹角最小者；
    //          平面2为剩余中与相机系+X轴夹角最小者；平面1为剩余者
    int plane1_idx = -1, plane2_idx = -1, plane3_idx = -1;
    const Eigen::Vector3f axis_neg_y(0.0f, -1.0f, 0.0f);
    const Eigen::Vector3f axis_pos_x(1.0f, 0.0f, 0.0f);
    float best_cos_to_neg_y = -std::numeric_limits<float>::max();
    float best_cos_to_pos_x = -std::numeric_limits<float>::max();

    for(int i = 0; i < 3; i++)
    {
        Eigen::Vector3f n = cur_boards[i].plane.head<3>();
        float n_norm = n.norm();
        if(n_norm < 1e-9f) continue;
        float cos_to_neg_y = n.dot(axis_neg_y) / n_norm;
        if(cos_to_neg_y > best_cos_to_neg_y)
        {
            best_cos_to_neg_y = cos_to_neg_y;
            plane3_idx = i;
        }
    }

    for(int i = 0; i < 3; i++)
    {
        if(i == plane3_idx) continue;
        Eigen::Vector3f n = cur_boards[i].plane.head<3>();
        float n_norm = n.norm();
        if(n_norm < 1e-9f) continue;
        float cos_to_pos_x = n.dot(axis_pos_x) / n_norm;
        if(cos_to_pos_x > best_cos_to_pos_x)
        {
            best_cos_to_pos_x = cos_to_pos_x;
            plane2_idx = i;
        }
    }
    for(int i = 0; i < 3; i++)
    {
        if(i != plane2_idx && i != plane3_idx)
        {
            plane1_idx = i;
            break;
        }
    }

    if(plane1_idx < 0 || plane2_idx < 0 || plane3_idx < 0)
    {
        return false;
    }

    // 根据排序顺序重新组织标定板数据
    struct BOARD board0 = cur_boards[plane1_idx];
    struct BOARD board1 = cur_boards[plane2_idx];
    struct BOARD board2 = cur_boards[plane3_idx];
    // cout<<"原始板 "<<plane1_idx<<"变成新板0"<<endl;
    // cout<<"原始板 "<<plane2_idx<<"变成新板1"<<endl;
    // cout<<"原始板 "<<plane3_idx<<"变成新板2"<<endl;
    // board0.plane = -board0.plane;//实验用！！！！！
    // board1.plane = -board1.plane;//实验用！！！！！
    // board2.plane = -board2.plane;//实验用！！！！！
    cur_boards[0] = board0;
    cur_boards[1] = board1;
    cur_boards[2] = board2;
    
    // 计算排序后三个平面的法向夹角，并判断是否都大于80度
    bool angle_valid = compute_plane_angle();
    
    // line_plane_pairs[i] = {a, b}: line_u为plane[i]与plane[a]交线，line_v为plane[i]与plane[b]交线
    compute_line_model(line_plane_pairs_);
    return angle_valid;
}


// 对角点进行排序，生成 orderd_corners 和 orderd_corners_3d
void Camera::sort_corners()
{
    // 找三平面的原点,使三点组成的三角形周长最小
    float min_perimeter = std::numeric_limits<float>::max();
    std::array<int, 3> origin_indices = {0, 0, 0};

    for(size_t i0 = 0; i0 < cur_boards[0].corners_3d.size(); i0++)
    {
        for(size_t i1 = 0; i1 < cur_boards[1].corners_3d.size(); i1++)
        {
            for(size_t i2 = 0; i2 < cur_boards[2].corners_3d.size(); i2++)
            {
                cv::Point3f p0 = cur_boards[0].corners_3d[i0];
                cv::Point3f p1 = cur_boards[1].corners_3d[i1];
                cv::Point3f p2 = cur_boards[2].corners_3d[i2];

                float d01 = std::sqrt((p0.x-p1.x)*(p0.x-p1.x) + (p0.y-p1.y)*(p0.y-p1.y) + (p0.z-p1.z)*(p0.z-p1.z));
                float d12 = std::sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
                float d20 = std::sqrt((p2.x-p0.x)*(p2.x-p0.x) + (p2.y-p0.y)*(p2.y-p0.y) + (p2.z-p0.z)*(p2.z-p0.z));

                float perimeter = d01 + d12 + d20;
                if(perimeter < min_perimeter)
                {
                    min_perimeter = perimeter;
                    origin_indices[0] = i0;
                    origin_indices[1] = i1;
                    origin_indices[2] = i2;
                }
            }
        }
    }

    // 设置各平面的原点
    for(int i = 0; i < 3; i++)
    {
        cv::Point3f pt = cur_boards[i].corners_3d[origin_indices[i]];
        cur_boards[i].origin_2d.push_back(cur_boards[i].corners[origin_indices[i]]);
        cur_boards[i].origin = Eigen::Vector3f(pt.x, pt.y, pt.z);
    }

    // 对每个平面进行角点排序
    for(int board_idx = 0; board_idx < 3; board_idx++)
    {
        const std::vector<cv::Point3f>& pts_3d = cur_boards[board_idx].corners_3d;
        const std::vector<cv::Point2f>& pts_2d = cur_boards[board_idx].corners;
        Eigen::Vector3f origin = cur_boards[board_idx].origin;
        Eigen::Vector3f line_u = cur_boards[board_idx].line_u;
        Eigen::Vector3f line_v = cur_boards[board_idx].line_v;

        int origin_idx = origin_indices[board_idx];

        // 用于记录哪些点已被使用
        std::vector<bool> used(pts_3d.size(), false);
        used[origin_idx] = true;

        // 行起点列表（第一个是 origin）
        std::vector<int> row_start_indices;
        row_start_indices.push_back(origin_idx);

        // 沿 line_v 方向找 corners_row-1 个行起点
        for(int r = 0; r < corners_row - 1; r++)
        {
            float min_dist = std::numeric_limits<float>::max();
            int best_idx = -1;

            for(size_t j = 0; j < pts_3d.size(); j++)
            {
                if(used[j]) continue;

                Eigen::Vector3f pt(pts_3d[j].x, pts_3d[j].y, pts_3d[j].z);
                Eigen::Vector3f v = pt - origin;
                float proj = v.dot(line_v);
                if(proj <= 0) continue;  // 只考虑 line_v 正方向

                // 点到射线的垂直距离
                Eigen::Vector3f perpendicular = v - proj * line_v;
                float dist = perpendicular.norm();

                if(dist < min_dist)
                {
                    min_dist = dist;
                    best_idx = j;
                }
            }

            if(best_idx >= 0)
            {
                row_start_indices.push_back(best_idx);
                used[best_idx] = true;
            }
        }

        // 按距离原点的距离排序行起点（由近到远）
        std::sort(row_start_indices.begin(), row_start_indices.end(), [&](int a, int b){
            Eigen::Vector3f pa(pts_3d[a].x, pts_3d[a].y, pts_3d[a].z);
            Eigen::Vector3f pb(pts_3d[b].x, pts_3d[b].y, pts_3d[b].z);
            return (pa - origin).norm() < (pb - origin).norm();
        });

        // 对每个行起点，沿 line_u 方向找 corners_col-1 个点
        std::vector<std::vector<int>> rows;
        for(size_t r = 0; r < row_start_indices.size(); r++)
        {
            std::vector<int> row;
            row.push_back(row_start_indices[r]);

            Eigen::Vector3f row_origin(pts_3d[row_start_indices[r]].x,
                                       pts_3d[row_start_indices[r]].y,
                                       pts_3d[row_start_indices[r]].z);

            for(int c = 0; c < corners_col - 1; c++)
            {
                float min_dist = std::numeric_limits<float>::max();
                int best_idx = -1;

                for(size_t j = 0; j < pts_3d.size(); j++)
                {
                    if(used[j]) continue;

                    Eigen::Vector3f pt(pts_3d[j].x, pts_3d[j].y, pts_3d[j].z);
                    Eigen::Vector3f v = pt - row_origin;
                    float proj = v.dot(line_u);
                    if(proj <= 0) continue;  // 只考虑 line_u 正方向

                    // 点到射线的垂直距离
                    Eigen::Vector3f perpendicular = v - proj * line_u;
                    float dist = perpendicular.norm();

                    if(dist < min_dist)
                    {
                        min_dist = dist;
                        best_idx = j;
                    }
                }

                if(best_idx >= 0)
                {
                    row.push_back(best_idx);
                    used[best_idx] = true;
                }
            }

            // 按距离行起点的距离排序（行起点除外）
            if(row.size() > 1)
            {
                std::sort(row.begin() + 1, row.end(), [&](int a, int b){
                    Eigen::Vector3f pa(pts_3d[a].x, pts_3d[a].y, pts_3d[a].z);
                    Eigen::Vector3f pb(pts_3d[b].x, pts_3d[b].y, pts_3d[b].z);
                    return (pa - row_origin).norm() < (pb - row_origin).norm();
                });
            }

            rows.push_back(row);
        }

        // 拼接成 orderd_corners_3d 和 orderd_corners
        for(const auto& row : rows)
        {
            for(int idx : row)
            {
                cur_boards[board_idx].orderd_corners_3d.push_back(pts_3d[idx]);
                cur_boards[board_idx].orderd_corners.push_back(pts_2d[idx]);
            }
        }
    }
}

 bool Camera::add(string path)
{
    org_img=cv::imread(path,1);
    if(org_img.empty())
    {
        cout << "加载图像失败，路径无效或文件为空: " << path << endl;
        return false;
    }
    img = org_img.clone();
    img_indx++;
    cv::putText(img,std::to_string(img_indx),cv::Point2f(10,30),cv::FONT_HERSHEY_SIMPLEX,0.7,cv::Scalar(155,155,155),3);

    if(!initialization)
        init_img();

    DataClear();

    CornerDetAC corner_detector(img);
    ChessboradStruct chessboardstruct;
    // 检测棋盘格角点与棋盘格
    cout <<endl<< "------------------------------------------------"<< endl;
     cout << "camera_frame: "<< path << endl;
     corner_detector.detectCorners(img, corners_p, corners_s, corner_detect_threshold);
    ImageChessesStruct ics;
    int target_corner_count = corners_col * corners_row;
    chessboardstruct.chessboardsFromCorners(corners_s, chessboards, chessboard_threshold, target_corner_count);
    cout << "detected chessboards num: " << chessboards.size() << endl;

    bool ischoose = false;
    // 排序
    if(Ensure_ValidFrame(chessboards))
    {
        // 提取棋盘格角点坐标
        extract_corners();
        // 为标定板排序，并判断三平面夹角是否都大于80度
        bool boards_valid = sort_boards();
        if(boards_valid)
        {
            // 对每个标定板的角点进行排序
            sort_corners();
            visualize_chessboards();
            visualize_corners();
            // visualize_masks();

            // 收集三个标定板的所有平面方程与有序角点
            vector<cv::Point2f> three_bd_2d_corners;
            vector<cv::Point3f> three_bd_3d_corners;
            vector<Eigen::Vector4f> three_bd_planes;
            for(int indx_bd = 0;indx_bd<3;indx_bd++)
            {
                for(int i=0;i<cur_boards[indx_bd].orderd_corners.size();i++)
                    three_bd_2d_corners.push_back(cur_boards[indx_bd].orderd_corners[i]);
                for(int i=0;i<cur_boards[indx_bd].orderd_corners_3d.size();i++)
                    three_bd_3d_corners.push_back(cur_boards[indx_bd].orderd_corners_3d[i]);
                three_bd_planes.push_back(cur_boards[indx_bd].plane);
            }

            all_2d_corners.insert(pair<int,vector<Point2f>>(img_indx,three_bd_2d_corners));
            all_3d_corners.insert(pair<int,vector<Point3f>>(img_indx,three_bd_3d_corners));
            cam_planes.insert(pair<int,vector<Eigen::Vector4f>>(img_indx,three_bd_planes));

            camera_cal_frame.push_back(img_indx);
            ischoose = true;
        }
        else
        {
            cout << ">>> 帧 " << img_indx << " 为无效帧（三平面夹角存在<=80度）" << endl;
        }
    }
    else
    {
        cout << ">>> 帧 " << img_indx << " 为无效帧" << endl;
    }
    return ischoose;
}


void Camera::GetIntrincMatrix(cv::Mat &intrincMatrix_)
{
    intrincMatrix_=intrincMatrix.clone();
}

void Camera::GetDistParameter(cv::Mat &distParameter_)
{
    distParameter_=distParameter.clone();
}
void Camera::Get2Dpoint(map<int,vector<cv::Point2f>> &all_corners_)
{
    all_corners_=all_2d_corners;
}

void Camera::Get3Dpoint(map<int, vector<cv::Point3f>> &all_corners_) {
    all_corners_ = all_3d_corners;
}

void Camera::GetPlanesModels(map<int,vector<Eigen::Vector4f>> &cam_planes_)
{
    cam_planes_=cam_planes;
}

}


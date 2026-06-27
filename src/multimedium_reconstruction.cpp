// multimedium_reconstruction.cpp
#include "multimedium_reconstruction.h"

MultimediumReconstruction::MultimediumReconstruction() {
}

MultimediumReconstruction::~MultimediumReconstruction() {
}

// ========== 构造函数，初始化所有多介质重建参数 ==========
MultimediumReconstruction::MultimediumReconstruction(
    const cv::Mat& camera_matrix_left,
    const cv::Mat& dist_coeffs_left,
    const cv::Mat& camera_matrix_right,
    const cv::Mat& dist_coeffs_right,
    const Eigen::Matrix3d& R_right_to_left,
    const Eigen::Vector3d& t_right_to_left,
    double d0_left,
    double d0_right,
    const Eigen::Vector3d& axis_left,
    const Eigen::Vector3d& axis_right,
    double d1,
    double mu0,
    double mu1,
    double mu2,
    const cv::Rect& roi_rect_left,
    const cv::Rect& roi_rect_right)
    : camera_matrix_left_(camera_matrix_left.clone()),
      dist_coeffs_left_(dist_coeffs_left.clone()),
      camera_matrix_right_(camera_matrix_right.clone()),
      dist_coeffs_right_(dist_coeffs_right.clone()),

      // 初始化列表：相机外参
      R_right_to_left_(R_right_to_left),
      t_right_to_left_(t_right_to_left),

      // 初始化列表：介质几何参数
      d0_left_(d0_left),
      d0_right_(d0_right),
      axis_left_(axis_left.normalized()),    // 归一化
      axis_right_(axis_right.normalized()),  // 归一化
      d1_(d1),

      // 初始化列表：折射率参数
      mu0_(mu0),
      mu1_(mu1),
      mu2_(mu2),

      // 初始化列表：图像处理参数
      roi_rect_left_(roi_rect_left),
      roi_rect_right_(roi_rect_right) 
{
    // 参数验证
    bool valid = true;

    // 验证相机内参矩阵
    if (camera_matrix_left_.rows != 3 || camera_matrix_left_.cols != 3) {
        std::cerr << "Error: Left camera matrix must be 3x3" << std::endl;
        valid = false;
    }

    if (camera_matrix_right_.rows != 3 || camera_matrix_right_.cols != 3) {
        std::cerr << "Error: Right camera matrix must be 3x3" << std::endl;
        valid = false;
    }

    // 验证畸变系数
    if (dist_coeffs_left_.rows != 5 || dist_coeffs_left_.cols != 1) {
        std::cerr << "Error: Left distortion coefficients must be 5x1" << std::endl;
        valid = false;
    }

    if (dist_coeffs_right_.rows != 5 || dist_coeffs_right_.cols != 1) {
        std::cerr << "Error: Right distortion coefficients must be 5x1" << std::endl;
        valid = false;
    }

    // 验证旋转矩阵正交性
    Eigen::Matrix3d I = R_right_to_left_ * R_right_to_left_.transpose();
    double ortho_error = (I - Eigen::Matrix3d::Identity()).norm();
    if (ortho_error > 1e-6) {
        std::cerr << "Warning: Rotation matrix may not be orthogonal. Error: "
                  << ortho_error << std::endl;
    }

    // 验证几何参数
    if (d0_left_ <= 0 || d0_right_ <= 0 || d1_ <= 0) {
        std::cerr << "Error: Distance parameters must be positive" << std::endl;
        valid = false;
    }

    // 验证折射率参数
    if (mu0_ <= 0 || mu1_ <= 0 || mu2_ <= 0) {
        std::cerr << "Error: Refractive indices must be positive" << std::endl;
        valid = false;
    }

    // 验证折射轴方向
    if (std::abs(axis_left_.norm() - 1.0) > 1e-6) {
        std::cerr << "Warning: Left axis vector is not normalized. Normalizing..." << std::endl;
        axis_left_.normalize();
    }

    if (std::abs(axis_right_.norm() - 1.0) > 1e-6) {
        std::cerr << "Warning: Right axis vector is not normalized. Normalizing..." << std::endl;
        axis_right_.normalize();
    }

    // 验证ROI区域
    if (roi_rect_left_.width <= 0 || roi_rect_left_.height <= 0 ||
        roi_rect_right_.width <= 0 || roi_rect_right_.height <= 0) {
        std::cerr << "Error: ROI dimensions must be positive" << std::endl;
        valid = false;
    }



}

// 辅助函数：从文件名中提取数字（如 "21.bmp" → 21）
int extractNumberFromFilename(const std::string& filepath) {
    // 1. 提取文件名（从最后一个 '/' 后截取）
    size_t slash_pos = filepath.find_last_of('/');
    std::string filename = (slash_pos == std::string::npos) ? filepath : filepath.substr(slash_pos + 1);

    // 2. 提取文件名中的纯数字部分
    std::string num_str;
    for (char c : filename) {
        if (isdigit(c)) {  // 只保留数字字符
            num_str += c;
        }
    }

    // 3. 转换为整数（无数字则返回0）
    if (num_str.empty())
        return 0;
    return std::stoi(num_str);
}

// 加载文件路径
std::vector<std::string> loadImagePaths(const std::string& foldername) {
    std::vector<std::string> image_paths;
    image_paths.clear();
    DIR* dir = opendir(foldername.c_str());
    if (!dir)
        return image_paths;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) {
            image_paths.push_back(foldername + "/" + entry->d_name);
        }
    }
    closedir(dir);

    // ========== 关键：按文件名中的数字升序排序 ==========
    std::sort(image_paths.begin(), image_paths.end(),
              [](const std::string& a, const std::string& b) {
                  int num_a = extractNumberFromFilename(a);
                  int num_b = extractNumberFromFilename(b);
                  return num_a < num_b;  // 升序（1,2,3...25），降序则改为 num_a > num_b
              });
    // image_nums_ = image_paths.size();

    return image_paths;
}

// ========== 重建对应的左右相机图像(单张光顺+暴力遍历) ==========
void MultimediumReconstruction::reconstructImageSmooth(const cv::Mat& image_left,
                                                 const cv::Mat& image_right,
                                                 std::vector<Eigen::Vector3d>& image_points) 
{
    Eigen::Vector3d mu(mu0_, mu1_, mu2_);

    // 1. 激光条纹中心线提取
    // 左相机
    std::vector<Eigen::Vector2d> image_points_left;
    cv::Mat imageline_left;
    LogMedialCenterLineExtractionNew(image_left, roi_rect_left_, image_points_left, imageline_left);

    // 右相机
    std::vector<Eigen::Vector2d> image_points_right;
    LogMedialCenterLineExtractionNew(image_right, roi_rect_right_, image_points_right, imageline_right);
    if(image_points_left.empty()||image_points_right.empty())
    {
        return ;
    }

    // 2. 对激光条纹进行光顺处理
    // 2.1 设置光顺参数
    double control_threshold = 0.05;                         // 控制点比例
    int num_sample_points = image_points_left.size();       // 重新采样点数

    // 2.2 获取拟合的曲线参数 
    int num_control_points_left = image_points_left.size() * control_threshold; // 控制点数
    int num_control_points_right = image_points_right.size() * control_threshold; // 控制点数
    BSplineCurve curve_left = fit_bspline_curve(image_points_left, num_control_points_left);
    BSplineCurve curve_right = fit_bspline_curve(image_points_right, num_control_points_right);

    // 2.3 重新生成光顺点集
    std::vector<Eigen::Vector2d> fitted_curve_left(num_sample_points);
    std::vector<Eigen::Vector2d> fitted_curve_right(num_sample_points);
    std::vector<double> fitted_points_right_index(num_sample_points);

    // 并行生成光顺点集（每个点独立计算，无数据依赖）
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_sample_points; ++i) {
        double t = static_cast<double>(i) / (num_sample_points - 1);
        if (t > 1.0) t = 1.0; // 强制钳位

        fitted_curve_left[i]  = calculate_bspline_point(curve_left,  t);
        fitted_curve_right[i] = calculate_bspline_point(curve_right, t);
        fitted_points_right_index[i] = t;
    }
    image_points_left  = fitted_curve_left;
    image_points_right = fitted_curve_right;


    // 提前读取相机内参
    double fx_left = camera_matrix_left_.at<double>(0, 0);
    double fy_left = camera_matrix_left_.at<double>(1, 1);
    double cx_left = camera_matrix_left_.at<double>(0, 2);
    double cy_left = camera_matrix_left_.at<double>(1, 2);

    double fx_right = camera_matrix_right_.at<double>(0, 0);
    double fy_right = camera_matrix_right_.at<double>(1, 1);
    double cx_right = camera_matrix_right_.at<double>(0, 2);
    double cy_right = camera_matrix_right_.at<double>(1, 2);

    int n_left  = static_cast<int>(image_points_left.size());
    int n_right = static_cast<int>(image_points_right.size());

    // 2. 像素点转反向投影光线（结合相机内参）+ 计算相机水界面出射点
    // 合并为一个并行循环，减少循环开销
    std::vector<Eigen::Vector3d> water_points_left(n_left);
    std::vector<Eigen::Vector3d> water_rays_left(n_left);

    // 并行计算左相机反投影光线和折射出射点
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_left; i++) {
        cv::Point2f pixel;
        pixel.x = image_points_left[i](0);
        pixel.y = image_points_left[i](1);

        double fx_left_ = camera_matrix_left_.at<double>(0, 0);
        double fy_left_ = camera_matrix_left_.at<double>(1, 1);
        double cx_left_ = camera_matrix_left_.at<double>(0, 2);
        double cy_left_ = camera_matrix_left_.at<double>(1, 2);

        Eigen::Vector3d v0_ray_left = pixelToCamRay(pixel, fx_left, fy_left, cx_left, cy_left);

        backwardProjectionCase3(d0_left_, d1_, axis_left_, mu, v0_ray_left,
                                water_points_left[i], water_rays_left[i]);
    }

    // 计算右相机水界面出射点
    std::vector<Eigen::Vector3d> water_points_right(n_right);
    std::vector<Eigen::Vector3d> water_rays_right(n_right);

    // 并行计算右相机反投影光线和折射出射点，并转换到左相机视角下
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_right; i++) {
        cv::Point2f pixel;
        pixel.x = image_points_right[i](0);
        pixel.y = image_points_right[i](1);

        Eigen::Vector3d v0_ray_right = pixelToCamRay(pixel, fx_right, fy_right, cx_right, cy_right);

        Eigen::Vector3d wp, wr;
        backwardProjectionCase3(d0_right_, d1_, axis_right_, mu, v0_ray_right, wp, wr);

        // 转换到左相机视角下
        water_points_right[i] = R_right_to_left_.transpose() * (wp - t_right_to_left_);
        water_rays_right[i]   = R_right_to_left_.transpose() * wr;
    }

    // 5. 计算空间三维点
    image_points.clear();

    // 预分配结果容器，避免并行写 vector 竞争
    std::vector<Eigen::Vector3d> result_points(n_left, Eigen::Vector3d::Zero());
    std::vector<bool> result_valid(n_left, false);

    // 遍历左相机出水点（最耗时的 O(n*m) 循环，使用 dynamic 调度）
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < n_left; i++) {
        int right_index = -1;
        double min_res = std::numeric_limits<double>::max();

        // 遍历右相机出水点
        for (int j = 0; j < n_right; j++) {
            Eigen::Vector3d mid_point;
            double reprojection_error = 0;
            midPointReconstruction(water_points_left[i],
                                   water_rays_left[i],
                                   water_points_right[j],
                                   water_rays_right[j],
                                   mid_point,
                                   &reprojection_error);       

            if (reprojection_error < min_res) {
                min_res = reprojection_error;
                right_index = j;
            }
        }

        if (right_index < 0 || min_res == std::numeric_limits<double>::max()) {
            continue;
        }

        // 根据邻域大小进行分组
        int region = 10;
        int right_limit = static_cast<int>(right_index) - region;
        int left_limit  = static_cast<int>(right_index) + region;

        if(right_limit < 0 || left_limit >= n_right)
        {
            // 采用中点求交计算空间坐标
            Eigen::Vector3d mid_point;
            double reprojection_error = 0;
            midPointReconstruction(water_points_left[i],
                                water_rays_left[i],
                                water_points_right[right_index],
                                water_rays_right[right_index],
                                mid_point,
                                &reprojection_error);

            result_points[i] = mid_point;
            result_valid[i]  = true;

        }else{
            std::vector<ErrorPoint> error_data;
            double min_error = std::numeric_limits<double>::max();
            size_t rough_min_idx = 0;
            double u_optimal = fitted_points_right_index[right_index]; // 默认回退值

            // 取 right_index 左右邻域的点
            for (int k = right_limit; k <= left_limit; k++) {
                // 边界检查
                if (k < 0 || k >= n_right) continue;

                Eigen::Vector3d mid_point;
                double reprojection_error = 0;
                
                // 调用重构函数
                midPointReconstruction(water_points_left[i],
                                        water_rays_left[i],
                                        water_points_right[k],
                                        water_rays_right[k],
                                        mid_point,
                                        &reprojection_error);

                double current_u = fitted_points_right_index[k];
                error_data.push_back({current_u, reprojection_error});

                // 记录粗略的最小点位置，用于分割左右区间
                if (reprojection_error < min_error) {
                    min_error = reprojection_error;
                    rough_min_idx = error_data.size() - 1;
                }
            }

            // --- 2. 双线交点法计算 ---
            if (rough_min_idx > 5 && rough_min_idx < error_data.size() - 5) {
                std::vector<ErrorPoint> left_set, right_set;
                
                // 分割数据：左侧下降段和右侧上升段
                // 注意：为了避开底部可能的非线性拐点，可以稍微远离中心点
                for (size_t m = 0; m < rough_min_idx; ++m) left_set.push_back(error_data[m]);
                for (size_t m = rough_min_idx + 1; m < error_data.size(); ++m) right_set.push_back(error_data[m]);

                double k1, b1, k2, b2;
                linearFit(left_set, k1, b1);  // 拟合左线
                linearFit(right_set, k2, b2); // 拟合右线

                if (std::abs(k1 - k2) > 1e-9) {
                    u_optimal = (b2 - b1) / (k1 - k2);
                    double error_optimal = k1 * u_optimal + b1;
                }
            } else {
                // 边界情况，回退到粗搜索最优点
                Eigen::Vector3d mid_point;
                double reprojection_error = 0;
                midPointReconstruction(water_points_left[i],
                                       water_rays_left[i],
                                       water_points_right[right_index],
                                       water_rays_right[right_index],
                                       mid_point,
                                       &reprojection_error);
                result_points[i] = mid_point;
                result_valid[i]  = true;
                continue;
            }

            // 根据u_optimal计算右相机各参数
            Eigen::Vector2d pt_fit_right_optimal = calculate_bspline_point(curve_right, u_optimal);

            // 计算右相机光线
            Eigen::Vector3d v0_ray_right_pt_fit;
            cv::Point2f pixel;
            pixel.x = pt_fit_right_optimal(0);
            pixel.y = pt_fit_right_optimal(1);

            v0_ray_right_pt_fit = pixelToCamRay(pixel, fx_right, fy_right, cx_right, cy_right);
       
            Eigen::Vector3d water_points_right_pt_fit;
            Eigen::Vector3d water_rays_right_pt_fit;
            Eigen::Vector3d water_point_right;
            Eigen::Vector3d water_ray_right;

            backwardProjectionCase3(d0_right_, d1_, axis_right_, mu, v0_ray_right_pt_fit,
                                    water_point_right, water_ray_right);

            water_points_right_pt_fit = R_right_to_left_.transpose() * (water_point_right - t_right_to_left_);
            water_rays_right_pt_fit   = R_right_to_left_.transpose() *  water_ray_right;

            // 采用中点求交计算空间坐标
            Eigen::Vector3d mid_point;
            double reprojection_error = 0;
            midPointReconstruction(water_points_left[i],
                                water_rays_left[i],
                                water_points_right_pt_fit,
                                water_rays_right_pt_fit,
                                mid_point,
                                &reprojection_error);

            if (std::abs(reprojection_error) > 1) {
                continue;
            }
            result_points[i] = mid_point;
            result_valid[i]  = true;
        }

    }

    // 收集有效点（串行汇总，O(n)，代价极小）
    for (int i = 0; i < n_left; i++) {
        if (result_valid[i]) {
            image_points.push_back(result_points[i]);
        }
    }

    return;
}


// ========== 重建所有左右相机图像（多张） ==========
void MultimediumReconstruction::reconstructImages(void) 
{
    // 读取左右相机所有图像路径（排序）
    image_paths_left_.clear();
    image_paths_right_.clear();

    image_paths_left_ = loadImagePaths(images_folder_left_);
    image_paths_right_ = loadImagePaths(images_folder_right_);

    image_nums_ = image_paths_left_.size();

    cv::Mat image_left = cv::imread(image_paths_left_[0]);
    image_size_ = image_left.size();

    // 初始化映射表
    cv::initUndistortRectifyMap(camera_matrix_left_, dist_coeffs_left_, cv::Mat(), camera_matrix_left_,
                                image_size_, CV_32FC1,
                                map1_left, map2_left);

    cv::initUndistortRectifyMap(camera_matrix_right_, dist_coeffs_right_, cv::Mat(), camera_matrix_right_,
                                image_size_, CV_32FC1,
                                map1_right, map2_right);
    
    // 重建所有激光条纹
    std::vector<std::vector<Eigen::Vector3d>> points_images;
    for(size_t i = 0; i < image_nums_; i=i+1)
    {
        cv::Mat image_left = cv::imread(image_paths_left_[i]);
        cv::Mat image_right = cv::imread(image_paths_right_[i]);

        // 进行图像校正
        cv::Mat image_undistorted_left;
        cv::Mat image_undistorted_right;
        
        // 在循环中使用 remap
        cv::remap(image_left, image_undistorted_left, map1_left, map2_left, cv::INTER_LINEAR);
        cv::remap(image_right, image_undistorted_right, map1_right, map2_right, cv::INTER_LINEAR);

        // 重建单帧激光条纹
        std::vector<Eigen::Vector3d> points_image;
        reconstructImageSmooth(image_undistorted_left, image_undistorted_right, points_image);
        points_images.push_back(points_image);

        std::cout <<i<< "单帧图像点数" << points_image.size() << std::endl;
    }

    // 将所有点云导入3D_points_
    points_.clear();
    for (size_t i = 0; i < points_images.size(); i++) {
        for (size_t j = 0; j < points_images[i].size(); j++) {
            points_.push_back(points_images[i][j]);
        }
    }
    std::cout << "点云导入完成 " << std::endl;

    return;
}

// 保存点云为txt格式
bool MultimediumReconstruction::savePointCloudToTxt(const std::vector<Eigen::Vector3d>& points, 
                         const std::string& filename)
{
    // 自动创建父目录（如果不存在）
    std::filesystem::path filepath(filename);
    std::filesystem::path parent_dir = filepath.parent_path();

    if (!parent_dir.empty() && !std::filesystem::exists(parent_dir)) {
        std::error_code ec;
        if (!std::filesystem::create_directories(parent_dir, ec)) {
            std::cerr << "无法创建目录: " << parent_dir << " 错误: " << ec.message() << std::endl;
            return false;
        }
        std::cout << "已创建目录: " << parent_dir << std::endl;
    }

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }
    std::cout << "文件打开成功" << std::endl;

    // 设置输出精度
    outfile << std::fixed << std::setprecision(6);

    // 写入点云数据
    for (const auto& point : points) {
        outfile << point(0) << " "
                << point(1) << " "
                << point(2) << "\n";
    }

    outfile.close();
    std::cout << "成功保存 " << points.size() << " 个点到 " << filename << std::endl;
    return true;
}


// 保存点云为txt格式(2D)
bool MultimediumReconstruction::savePointCloudToTxt2D(const std::vector<Eigen::Vector2d> points, 
                                                    const std::string filename) 
{
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    // 设置输出精度
    outfile << std::fixed << std::setprecision(6);

    // 写入点云数据，每行格式: auto& point : points)
    for (size_t i = 0; i < points.size(); i++) {
        Eigen::Vector2d point = points[i];
        outfile << point(0) << " "
                << point(1) << std::endl;
    }

    outfile.close();
    std::cout << "成功保存 " << points.size() << " 个点到 " << filename << std::endl;
    return true;
}
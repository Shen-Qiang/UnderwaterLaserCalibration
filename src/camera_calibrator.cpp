// camera_calibrator.cpp
#include "camera_calibrator.h"
#include <dirent.h>
#include <iostream>

// 构造函数初始化
CameraCalibrator::CameraCalibrator(cv::Size board_size, float square_size)
    : board_size_(board_size),
      square_size_(square_size),
      total_corners_(board_size.width * board_size.height),
      reproj_error_(0.0) {
    // 初始化内参矩阵（默认值）
    camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
    // 初始化畸变系数（5参数：k1,k2,p1,p2,k3）
    dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
}

// 生成世界坐标系3D点（棋盘格平面为z=0）
void CameraCalibrator::generateObjectPoints(std::vector<cv::Point3f>& obj_points) {
    obj_points.clear();
    // 按行优先顺序生成3D点（x,y,z），z=0
    for (int y = 0; y < board_size_.height; ++y) {
        for (int x = 0; x < board_size_.width; ++x) {
            obj_points.emplace_back(
                x * square_size_,  // x坐标（列方向）
                y * square_size_,  // y坐标（行方向）
                0.0f               // z坐标（平面）
            );
        }
    }
}

// 添加图像并提取角点
bool CameraCalibrator::addImage(const cv::Mat& img) {
    if (img.empty()) {
        std::cerr << "错误：输入图像为空！" << std::endl;
        return false;
    }

    img_size_ = img.size();

    // 转换为灰度图（calib3d角点检测需要灰度图）
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 1) {
        gray = img.clone();
    } else {
        std::cerr << "错误：图像通道数不支持（仅支持1或3通道）！" << std::endl;
        return false;
    }

    // // 1. 粗检测角点（calib3d的findChessboardCorners）
    // std::vector<cv::Point2f> corners;
    // bool found = cv::findChessboardCorners(
    //     gray,
    //     board_size_,
    //     corners,
    //     cv::CALIB_CB_ADAPTIVE_THRESH |  // 自适应阈值
    //         cv::CALIB_CB_NORMALIZE_IMAGE);

    // // if (!found || corners.size() != total_corners_) {
    // if (!found) {
    //     std::cerr << "警告：未检测到完整的棋盘格角点（跳过该图像）！" << std::endl;
    //     return false;
    // }

    // 1. 使用 SB 版本检测角点（此方法对模糊和水下图像极其有效）
    std::vector<cv::Point2f> corners;
    int flags = cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY; 
    
    // SB 方法内部自带高精度亚像素提取，通常比 cornerSubPix 效果更好
    bool found = cv::findChessboardCornersSB(
        gray,
        board_size_,
        corners,
        flags);

    if (!found) {
        std::cerr << "警告：未检测到完整的棋盘格角点（跳过该图像）！" << std::endl;
        return false;
    }

    // 2. 亚像素精确化（calib3d的cornerSubPix）
    cv::cornerSubPix(
        gray,
        corners,
        cv::Size(11, 11),  // 搜索窗口大小
        cv::Size(-1, -1),  // 死区（无死区）
        cv::TermCriteria(  // 终止条件：30次迭代或0.001像素精度
            cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
            30, 0.001));

    // 3. 生成对应世界坐标并保存
    std::vector<cv::Point3f> obj_points;
    generateObjectPoints(obj_points);
    object_points_.push_back(obj_points);  // 世界坐标（固定）
    image_points_.push_back(corners);      // 图像坐标（检测结果）

    // 可视化角点（可选）
    // cv::Mat img_with_corners = img.clone();
    // cv::drawChessboardCorners(img_with_corners, board_size_, corners, found);
    // cv::namedWindow("角点检测结果", cv::WINDOW_NORMAL);
    // cv::imshow("角点检测结果", img_with_corners);
    // cv::waitKey(500);  // 显示500ms

    std::cout << "成功添加图像，累计 " << image_points_.size() << " 张有效图像" << std::endl;
    return true;
}

// 执行标定（核心：calib3d的calibrateCamera）
bool CameraCalibrator::calibrate() {
    if (image_points_.size() < 3) {  // 至少需要3张图像才能标定
        std::cerr << "错误：有效图像不足（至少需要3张）！" << std::endl;
        return false;
    }

    std::cout << object_points_.size() << std::endl;
    std::cout << image_points_.size() << std::endl;

    // 调用calib3d模块的相机标定函数
    reproj_error_ = cv::calibrateCamera(
        object_points_,  // 世界坐标系3D点集合
        image_points_,   // 图像坐标系2D点集合
        img_size_,       // 图像大小
        camera_matrix_,  // 输出内参矩阵
        dist_coeffs_,    // 输出畸变系数
        rvecs_,          // 输出旋转向量（外参）
        tvecs_,          // 输出平移向量（外参）
        0                // 标定 flags（默认即可）
    );

    // 输出标定结果
    std::cout << "\n===== 标定结果 =====" << std::endl;
    std::cout << "内参矩阵 (camera matrix):\n"
              << camera_matrix_ << std::endl
              << std::endl;
    std::cout << "畸变系数 (distortion coefficients):\n"
              << dist_coeffs_ << std::endl
              << std::endl;
    std::cout << "平均重投影误差: " << reproj_error_ << " 像素" << std::endl;
    std::cout << "======================" << std::endl;

    // 重投影误差越小越好（通常<1像素为优）
    return reproj_error_ < 1.0;  // 简单判断标定是否合格
}

// 畸变校正（calib3d的undistort）
void CameraCalibrator::undistortImage(const cv::Mat& distorted_img, cv::Mat& undistorted_img) {
    if (camera_matrix_.empty() || dist_coeffs_.empty()) {
        std::cerr << "错误：未进行标定或标定结果无效！" << std::endl;
        return;
    }
    // 使用calib3d的undistort函数校正畸变
    cv::undistort(distorted_img, undistorted_img, camera_matrix_, dist_coeffs_);
}

// 保存标定结果到文件
bool CameraCalibrator::saveCalibration(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "错误：无法打开文件 " << filename << " 进行写入！" << std::endl;
        return false;
    }

    // 保存相机参数
    fs << "camera_matrix" << camera_matrix_;
    fs << "dist_coeffs" << dist_coeffs_;
    fs << "reprojection_error" << reproj_error_;
    fs << "board_size" << board_size_;
    fs << "square_size" << square_size_;

    fs.release();
    std::cout << "标定结果已保存到 " << filename << std::endl;
    return true;
}

// 加载标定结果
bool CameraCalibrator::loadCalibration(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "错误：无法打开文件 " << filename << " 进行读取！" << std::endl;
        return false;
    }

    // 读取参数
    fs["camera_matrix"] >> camera_matrix_;
    fs["dist_coeffs"] >> dist_coeffs_;
    fs["reprojection_error"] >> reproj_error_;
    fs["board_size"] >> board_size_;
    fs["square_size"] >> square_size_;

    total_corners_ = board_size_.width * board_size_.height;
    fs.release();
    std::cout << "已从 " << filename << " 加载标定结果" << std::endl;
    return true;
}

int CameraCalibrator::loadImagePaths(const std::string& foldername) {
    image_paths_.clear();
    DIR* dir = opendir(foldername.c_str());
    if (!dir)
        return -1;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) {
            image_paths_.push_back(foldername + "/" + entry->d_name);
        }
    }
    closedir(dir);

    return image_paths_.size();
}

int CameraCalibrator::loadImages() {
    for (const auto& path : image_paths_) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "无法读取图像：" << path << std::endl;
            continue;
        }
        addImage(img);  // 提取角点并添加
    }

    return 0;
}

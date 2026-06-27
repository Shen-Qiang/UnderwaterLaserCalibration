// stereo_calibrator.cpp

#include "stereo_calibrator.h"

// ========== 构造函数 ==========
StereoCalibrator::StereoCalibrator(cv::Size board_size, float square_size)
    : board_size_(board_size),
      square_size_(square_size) {
    // 初始化参数
}

// ========== 析构函数 ==========
StereoCalibrator::~StereoCalibrator() {
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

// ========== 准备物体点坐标 ==========
std::vector<cv::Point3f> StereoCalibrator::prepareObjectPoints() const {
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < board_size_.height; i++) {
        for (int j = 0; j < board_size_.width; j++) {
            objp.push_back(cv::Point3f(j * square_size_, i * square_size_, 0));
        }
    }
    return objp;
}

// ========== 检测棋盘格角点 ==========
bool StereoCalibrator::detectChessboard(const cv::Mat& image, std::vector<cv::Point2f>& corners) const {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    bool found = cv::findChessboardCorners(gray, board_size_, corners,
                                           cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    if (found) {
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                          50, 0.001));
    }

    return found;
}

// ========== 添加标定图像对 ==========
bool StereoCalibrator::addImagePair(const cv::Mat& img_left, const cv::Mat& img_right, bool visualize) {
    if (img_left.empty() || img_right.empty()) {
        std::cerr << "错误：图像为空！" << std::endl;
        return false;
    }

    if (img_left.size() != img_right.size()) {
        std::cerr << "错误：左右图像尺寸不一致！" << std::endl;
        return false;
    }

    // 记录图像尺寸
    if (img_size_.width == 0) {
        img_size_ = img_left.size();
    }

    // 检测角点
    std::vector<cv::Point2f> corners_left, corners_right;
    bool found_left = detectChessboard(img_left, corners_left);
    bool found_right = detectChessboard(img_right, corners_right);

    if (found_left && found_right) {
        objpoints_.push_back(prepareObjectPoints());
        imgpoints_left_.push_back(corners_left);
        imgpoints_right_.push_back(corners_right);

        if (visualize) {
            cv::Mat vis_left = img_left.clone();
            cv::Mat vis_right = img_right.clone();
            drawChessboardCorners(vis_left, board_size_, corners_left, found_left);
            drawChessboardCorners(vis_right, board_size_, corners_right, found_right);

            cv::imshow("Left", vis_left);
            cv::imshow("Right", vis_right);
            cv::waitKey(100);
        }

        std::cout << "成功添加第 " << objpoints_.size() << " 组图像" << std::endl;
        return true;
    } else {
        std::cout << "未检测到角点" << std::endl;
        return false;
    }
}

// ========== 从文件夹批量添加图像 ==========
int StereoCalibrator::addImagesFromFolder(const std::string& left_folder, const std::string& right_folder) {
    image_paths_left_.clear();
    image_paths_right_.clear();

    image_paths_left_ = loadImagePaths(left_folder);
    image_paths_right_ = loadImagePaths(right_folder);

    std::cout << "找到 " << image_paths_left_.size() << " 张左图像" << std::endl;
    std::cout << "找到 " << image_paths_right_.size() << " 张右图像" << std::endl;

    if (image_paths_left_.size() != image_paths_right_.size()) {
        std::cerr << "错误：左右图像数量不匹配！" << std::endl;
        return 0;
    }

    int success_count = 0;
    for (size_t i = 0; i < image_paths_left_.size(); i++) {
        cv::Mat img_left = cv::imread(image_paths_left_[i]);
        cv::Mat img_right = cv::imread(image_paths_right_[i]);

        if (addImagePair(img_left, img_right, false)) {
            success_count++;
        }
    }

    cv::destroyAllWindows();
    std::cout << "\n成功添加 " << success_count << " 组图像" << std::endl;
    return success_count;
}

// ========== 单目标定 ==========
bool StereoCalibrator::calibrateMonocular() {
    if (objpoints_.size() < 10) {
        std::cerr << "错误：图像数量不足（至少需要10组），当前: " << objpoints_.size() << std::endl;
        return false;
    }

    std::cout << "\n========== 左相机标定 ==========" << std::endl;
    std::vector<cv::Mat> rvecs_left, tvecs_left;
    rms_left_ = cv::calibrateCamera(objpoints_, imgpoints_left_, img_size_,
                                    camera_matrix_left_, dist_coeffs_left_, rvecs_left, tvecs_left);

    std::cout << "左相机重投影误差: " << rms_left_ << " 像素" << std::endl;

    std::cout << "========== 右相机标定 ==========" << std::endl;
    std::vector<cv::Mat> rvecs_right, tvecs_right;
    rms_right_ = cv::calibrateCamera(objpoints_, imgpoints_right_, img_size_,
                                     camera_matrix_right_, dist_coeffs_right_, rvecs_right, tvecs_right);

    std::cout << "右相机重投影误差: " << rms_right_ << " 像素" << std::endl;

    return true;
}

// ========== 双目标定 ==========
bool StereoCalibrator::calibrateStereo() {
    if (camera_matrix_left_.empty() || camera_matrix_right_.empty()) {
        std::cerr << "错误：请先执行单目标定！" << std::endl;
        return false;
    }
    std::cout << "========== 双目标定 ==========" << std::endl;

    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 1e-5);

    rms_stereo_ = stereoCalibrate(
        objpoints_, imgpoints_left_, imgpoints_right_,
        camera_matrix_left_, dist_coeffs_left_,
        camera_matrix_right_, dist_coeffs_right_,
        img_size_, R_, T_, E_, F_,
        cv::CALIB_FIX_INTRINSIC,
        criteria);

    baseline_ = norm(T_);

    std::cout << "双目标定重投影误差: " << rms_stereo_ << " 像素" << std::endl;
    std::cout << "基线距离: " << baseline_ << " mm" << std::endl;

    is_calibrated_ = true;
    return true;
}

// ========== 保存标定参数 ==========
bool StereoCalibrator::saveCalibration(const std::string& filename) const {
    if (!is_calibrated_) {
        std::cerr << "错误：尚未完成标定！" << std::endl;
        return false;
    }

    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "错误：无法打开文件 " << filename << std::endl;
        return false;
    }

    // 基本信息
    fs << "board_size_width" << board_size_.width;
    fs << "board_size_height" << board_size_.height;
    fs << "square_size" << square_size_;
    fs << "image_width" << img_size_.width;
    fs << "image_height" << img_size_.height;

    // 单目标定结果
    fs << "camera_matrix_left" << camera_matrix_left_;
    fs << "dist_coeffs_left" << dist_coeffs_left_;
    fs << "camera_matrix_right" << camera_matrix_right_;
    fs << "dist_coeffs_right" << dist_coeffs_right_;
    fs << "rms_left" << rms_left_;
    fs << "rms_right" << rms_right_;

    // 双目标定结果
    fs << "R" << R_;
    fs << "T" << T_;
    fs << "E" << E_;
    fs << "F" << F_;
    fs << "rms_stereo" << rms_stereo_;
    fs << "baseline" << baseline_;

    // // 立体校正结果
    // if(is_rectified_) {
    //     fs << "R1" << R1_;
    //     fs << "R2" << R2_;
    //     fs << "P1" << P1_;
    //     fs << "P2" << P2_;
    //     fs << "Q" << Q_;
    //     fs << "map_left_x" << map_left_x_;
    //     fs << "map_left_y" << map_left_y_;
    //     fs << "map_right_x" << map_right_x_;
    //     fs << "map_right_y" << map_right_y_;
    // }

    fs.release();
    std::cout << "标定参数已保存到: " << filename << std::endl;
    return true;
}

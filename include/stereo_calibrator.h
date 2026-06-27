#ifndef STEREO_CALIBRATOR_H
#define STEREO_CALIBRATOR_H

#include <string>
#include <vector>
#include <dirent.h>
#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// 双目相机标定
class StereoCalibrator {
   public:
    // ========== 构造函数 ==========
    /**
     * @brief 构造函数
     * @param board_size 棋盘格内角点数量 (列, 行)
     * @param square_size 方格实际大小 (mm)
     */
    StereoCalibrator(cv::Size board_size = cv::Size(8, 11), float square_size = 25.0f);

    /**
     * @brief 析构函数
     */
    ~StereoCalibrator();

    // ========== 标定流程 ==========
    /**
     * @brief 添加标定图像对
     * @param img_left 左相机图像
     * @param img_right 右相机图像
     * @param visualize 是否可视化检测结果
     * @return 是否成功检测到角点
     */
    bool addImagePair(const cv::Mat& img_left, const cv::Mat& img_right, bool visualize = false);

    /**
     * @brief 从文件夹批量添加图像
     * @param left_folder 左相机图像文件夹路径
     * @param right_folder 右相机图像文件夹路径
     * @return 成功添加的图像对数量
     */
    int addImagesFromFolder(const std::string& left_folder, const std::string& right_folder);

    /**
     * @brief 执行单目标定
     * @return 是否标定成功
     */
    bool calibrateMonocular();

    /**
     * @brief 执行双目标定
     * @return 是否标定成功
     */
    bool calibrateStereo();

    // ========== 参数保存/加载 ==========
    /**
     * @brief 保存标定参数到文件
     * @param filename 文件路径
     * @return 是否保存成功
     */
    bool saveCalibration(const std::string& filename = "stereo_calibration.yml") const;

    /**
     * @brief 从文件加载标定参数
     * @param filename 文件路径
     * @return 是否加载成功
     */
    bool loadCalibration(const std::string& filename = "stereo_calibration.yml");

   private:
    // ========== 标定板参数 ==========
    cv::Size board_size_;  // 棋盘格内角点数量
    float square_size_;    // 方格实际大小 (mm)

    // ========== 图像参数 ==========
    std::vector<std::string> image_paths_left_;
    std::vector<std::string> image_paths_right_;

    // ========== 标定数据 ==========
    std::vector<std::vector<cv::Point3f>> objpoints_;        // 3D物体点
    std::vector<std::vector<cv::Point2f>> imgpoints_left_;   // 左相机2D图像点
    std::vector<std::vector<cv::Point2f>> imgpoints_right_;  // 右相机2D图像点
    cv::Size img_size_;                                      // 图像尺寸

    // ========== 单目标定结果 ==========
    cv::Mat camera_matrix_left_;   // 左相机内参矩阵
    cv::Mat dist_coeffs_left_;     // 左相机畸变系数
    cv::Mat camera_matrix_right_;  // 右相机内参矩阵
    cv::Mat dist_coeffs_right_;    // 右相机畸变系数

    double rms_left_;   // 左相机重投影误差
    double rms_right_;  // 右相机重投影误差

    // ========== 双目标定结果 ==========
    cv::Mat R_;  // 旋转矩阵（右相机相对左相机）
    cv::Mat T_;  // 平移向量（右相机相对左相机）
    cv::Mat E_;  // 本质矩阵
    cv::Mat F_;  // 基础矩阵

    double rms_stereo_;  // 双目重投影误差
    double baseline_;    // 基线距离 (mm)

    // ========== 状态标志 ==========
    bool is_calibrated_;  // 是否已完成标定
    bool is_rectified_;   // 是否已完成校正

    // ========== 私有辅助函数 ==========
    /**
     * @brief 准备物体点坐标
     */
    std::vector<cv::Point3f> prepareObjectPoints() const;

    /**
     * @brief 检测单张图像的棋盘格角点
     */
    bool detectChessboard(const cv::Mat& image, std::vector<cv::Point2f>& corners) const;
};

#endif

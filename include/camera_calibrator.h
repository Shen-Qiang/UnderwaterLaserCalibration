#ifndef CAMERA_CALIBRATION_H
#define CAMERA_CALIBRATION_H

#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 单目相机标定
class CameraCalibrator {
   public:
    // 构造函数
    CameraCalibrator(cv::Size board_size, float square_size = 0.025f);

    // 添加图像并提取角点
    bool addImage(const cv::Mat& img);

    // 执行标定
    bool calibrate();

    // 校正畸变图像
    void undistortImage(const cv::Mat& distorted_img, cv::Mat& undistorted_img);

    // 保存标定结果
    bool saveCalibration(const std::string& filename);

    // 加载标定结果
    bool loadCalibration(const std::string& filename);

    // 获取标定结果（只读）
    const cv::Mat& getCameraMatrix() const { return camera_matrix_; }
    const cv::Mat& getDistCoeffs() const { return dist_coeffs_; }
    double getReprojectionError() const { return reproj_error_; }

    // 加载文件路径
    int loadImagePaths(const std::string& foldername);

    // 加载标定图像
    int loadImages();

   private:
    // 棋盘格参数
    cv::Size board_size_;
    float square_size_;  // 单格尺寸（米）
    int total_corners_;  // 总角点数量（width*height）

    // 图像数据
    cv::Size img_size_;
    std::vector<std::string> image_paths_;
    std::vector<cv::Mat> images_;

    // 标定数据
    std::vector<std::vector<cv::Point3f>> object_points_;  // 世界坐标系3D点
    std::vector<std::vector<cv::Point2f>> image_points_;   // 图像坐标系2D点
    std::vector<cv::Mat> rvecs_;                           // 旋转向量（每幅图像的外参）
    std::vector<cv::Mat> tvecs_;                           // 平移向量（每幅图像的外参）

    // 标定结果
    cv::Mat camera_matrix_;  // 内参矩阵 (3x3)
    cv::Mat dist_coeffs_;    // 畸变系数 (1x5)
    double reproj_error_;    // 重投影误差（评估标定精度）

    // 生成棋盘格的世界坐标系3D点（z=0，原点在左上角）
    void generateObjectPoints(std::vector<cv::Point3f>& obj_points);
};

#endif

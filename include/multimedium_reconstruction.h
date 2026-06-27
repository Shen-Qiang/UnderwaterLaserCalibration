#ifndef MULTIMEDIUM_RECONSTRUCTION_H
#define MULTIMEDIUM_RECONSTRUCTION_H

#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/SVD>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <omp.h>

#include "tools.h"
#include "fit_tools.h"
#include "laser_extractor.h"

// 空气-玻璃-水多介质三维重建
class MultimediumReconstruction {
   public:
    MultimediumReconstruction();

    ~MultimediumReconstruction();

    /**
     * @brief 构造函数，初始化所有多介质重建参数
     *
     * @param camera_matrix_left 左相机内参矩阵 (3x3 CV_64F)
     * @param dist_coeffs_left 左相机畸变系数 (5x1 CV_64F)
     * @param camera_matrix_right 右相机内参矩阵 (3x3 CV_64F)
     * @param dist_coeffs_right 右相机畸变系数 (5x1 CV_64F)
     * @param R_right_to_left 右相机到左相机的旋转矩阵
     * @param t_right_to_left 右相机到左相机的平移向量
     * @param d0_left 左相机介质界面到相机的距离
     * @param d0_right 右相机介质界面到相机的距离
     * @param axis_left 左相机折射轴方向向量
     * @param axis_right 右相机折射轴方向向量
     * @param d1 介质厚度
     * @param mu0 空气折射率
     * @param mu1 第一层介质折射率
     * @param mu2 第二层介质折射率
     * @param roi_rect_left 左图像ROI区域
     * @param roi_rect_right 右图像ROI区域
     */
    MultimediumReconstruction(
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
        const cv::Rect& roi_rect_right);

    /**
     * @brief 重建所有左右相机图像
     * @param
     */
    void reconstructImages(void);

    /**
     * @brief 重建对应的左右相机图像(单张光顺)
     * @param image_left_ 左相机图片
     * @param image_right_ 右相机图片
     * @param points_image_ 重建三维点
     */
    void reconstructImageSmooth(const cv::Mat& image_left,
                          const cv::Mat& image_right,
                          std::vector<Eigen::Vector3d>& image_points);                          

    /**
     * @brief 保存点云为txt格式
     * @param filename 保存文件名
     */
    bool savePointCloudToTxt(const std::vector<Eigen::Vector3d>& points, 
                         const std::string& filename);
    
    /**
     * @brief 保存点云为txt格式(2D)
     * @param filename 保存文件名
     */
    bool savePointCloudToTxt2D(const std::vector<Eigen::Vector2d> points, 
                            const std::string filename);

    std::string images_folder_left_;
    std::string images_folder_right_;

    // 相机内参
    cv::Mat camera_matrix_left_;   // 左相机内参矩阵
    cv::Mat dist_coeffs_left_;     // 左相机畸变系数
    cv::Mat camera_matrix_right_;  // 右相机内参矩阵
    cv::Mat dist_coeffs_right_;    // 右相机畸变系数

    // 左右相机相对位姿
    Eigen::Matrix3d R_right_to_left_;  // 旋转矩阵（右相机相对左相机）
    Eigen::Vector3d t_right_to_left_;  // 旋转矩阵（右相机相对左相机）

    // 折射参数
    double d0_left_;              // 左相机光心至介质界面距离
    double d0_right_;             // 右相机光心至介质界面距离
    Eigen::Vector3d axis_left_;   // 左相机轴向量
    Eigen::Vector3d axis_right_;  // 右相机轴向量
    double d1_;                   // 玻璃厚度
    double mu0_;                  // 空气折射率
    double mu1_;                  // 玻璃折射率
    double mu2_;                  // 水折射率

    cv::Rect roi_rect_left_;   // ROI区域
    cv::Rect roi_rect_right_;  // ROI区域

    cv::Size image_size_;   // 图像规格

    std::vector<Eigen::Vector3d> points_;  // 重建三维点

   private:
    // 图像参数
    int image_nums_;
    std::vector<std::string> image_paths_left_;
    std::vector<std::string> image_paths_right_;

    cv::Mat map1_left, map2_left, map1_right, map2_right;
};

#endif
#ifndef MULTIMEDIUM_CALIBRATOR_H
#define MULTIMEDIUM_CALIBRATOR_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#include <Eigen/KroneckerProduct>
#include <Eigen/Polynomials>

#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include <dirent.h>
#include <iostream>

#include "tools.h"


// 定义结构体，语义更清晰
struct EstimationResult {
    double d0;                    // 相机到玻璃的距离
    double residual;              // 残差
    Eigen::Vector3d axis;         // 玻璃法向
    Eigen::Matrix3d rotation;     // 旋转矩阵
    Eigen::Vector3d translation;  // 平移向量
    
    // 构造函数
    EstimationResult(
        double d0_,
        double residual_,
        const Eigen::Vector3d& axis_,
        const Eigen::Matrix3d& rotation_,
        const Eigen::Vector3d& translation_)
        : d0(d0_),
          residual(residual_),
          axis(axis_),
          rotation(rotation_),
          translation(translation_) {}
    
    // 默认构造函数
    EstimationResult() 
        : d0(0.0), 
          residual(0.0), 
          axis(Eigen::Vector3d::Zero()),
          rotation(Eigen::Matrix3d::Identity()),
          translation(Eigen::Vector3d::Zero()) {}
};

// 多介质折射标定（封闭解）
class MultimediumCalibrator {
   public:
    MultimediumCalibrator(cv::Size board_size, float square_size = 15.0f);

    // 轴估计-11点法
    int estimateAxisElevenPoints();

    // 轴估计-8点法
    int estimateAxisEightPoints();

    // 轴估计（单张图像）
    int estimateAxisEightPointsImage(int image_index,
                                double &d0_closed,
                                Eigen::Vector3d & axis_closed,
                                Eigen::Matrix3d &R_closed,
                                Eigen::Vector3d &t_closed,
                                double &res_closed);

    // 轴估计（多张图像）
    int estimateAxisEightPointsImages(void);

    // 距离估计
    int estimateThickness();

    int getObjectPoints(const std::vector<std::vector<cv::Point3f>>& object_points);

    int getImagePoints(const std::vector<std::vector<cv::Point2f>>& image_points);

    // 光心入射光线计算
    int calculateV0Rays();

    // 生成世界坐标系3D点
    void generateObjectPoints(std::vector<cv::Point3f>& obj_points);

    // 添加图像并提取角点
    bool addImage(const cv::Mat& img);

    // 加载文件路径
    int loadImagePaths(const std::string& foldername);

    // 加载标定图像
    int loadImages();

    //
    void estimateTzTwoLayerCase2KnownMu(int image_index,
                                        const Eigen::Matrix3d R,
                                        const Eigen::Vector3d n,
                                        const Eigen::Vector3d s,
                                        const std::vector<int> ids,
                                        double& d0,
                                        double& d1,
                                        Eigen::Vector3d& t);

    //
    void estimateTzTwoLayerCase3KnownMu(int image_index,
                                        const Eigen::Matrix3d R,
                                        const Eigen::Vector3d n,
                                        const Eigen::Vector3d s,
                                        const std::vector<int> ids,
                                        double& d0,
                                        double& d1,
                                        Eigen::Vector3d& t);

    void estimateTzTwoLayerCase3KnownMuD1(int image_index,
                                          const Eigen::Matrix3d R,
                                          const Eigen::Vector3d n,
                                          const Eigen::Vector3d s,
                                          const std::vector<int> ids,
                                          double& d0,
                                          Eigen::Vector3d& t,
                                          double& res);

    /**
    * @brief 根据E32恢复E33
    * @param E32 本质矩阵E的前两列
    * @note 存在多解情况
    */
    std::vector<Eigen::Matrix<double, 3, 3>> SolveLastColOfEmatrix(const Eigen::Matrix<double, 3, 2>& E32);

    /**
    * @brief 根据本质矩阵E矩阵恢复轴线
    * @param E 本质矩阵E
    * @note 
    */
    Eigen::Vector3d getAxisFromE(const Eigen::Matrix<double, 3, 3>& E);

    /**
    * @brief 根据本质矩阵E恢复旋转矩阵R，并与s进行组合
    * @param E 本质矩阵E
    * @param s Axt=[A]t
    * @param ne 轴向向量
    * @return 旋转矩阵和s的组合
    * @note 共有8组解，（R1 R2 -R1 -R2）x（s -s）
    */
    std::pair<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>>
    getRotationMatricesFromE(const Eigen::Matrix3d& E,
                             const Eigen::Vector3d& s);

    /**
    * @brief 计算重投影误差
    * @param axis 轴线向量
    * @param R 标定板坐标系到相机坐标系的旋转矩阵(RP+t)
    * @param t 标定板坐标系到相机坐标系的平移向量(RP+t)
    * @param d0 相机光心->（空气-玻璃界面）距离
    * @param d1 （空气-玻璃界面）距离->（玻璃-水界面）距离
    * @param id 图像id
    * @param error 重投影误差
    * @return void
    * @note 先根据封闭解计算的R和t计算相机坐标系下的角点坐标，再利用正向投影计算归一化平面坐标，
            最后计算归一化平面坐标误差（空气-玻璃-水介质）
    */
    void getReprojectionErrorCase3(const Eigen::Vector3d axis,
                            const Eigen::Matrix3d R,
                            const Eigen::Vector3d t,
                            const double d0,
                            const double d1,
                            const int id,
                            double &error);

    // 相机内参
    double fx_;
    double fy_;
    double cx_;
    double cy_;

    // 畸变参数
    double k1_;
    double k2_;
    double p1_;
    double p2_;
    double k3_;

    std::vector<std::vector<cv::Point3f>> object_points_;  // 世界坐标系3D点
    std::vector<std::vector<cv::Point2f>> image_points_;   // 图像坐标系2D点
    std::vector<std::vector<Eigen::Vector3d>> v0_rays_;    // 穿过图像角点光线
    std::vector<std::vector<std::vector<Eigen::Vector3d>>> v0_rays_3d_;

    // 封闭求解结果
    std::vector<double> d0_closed_;
    std::vector<Eigen::Vector3d> axis_closed_;
    std::vector<Eigen::Matrix3d> R_closed_;
    std::vector<Eigen::Vector3d> t_closed_;    

    int image_nums_;            // 图像数量


   private:
    int flag;               // 介质模式
    float thickness_;       //
    Eigen::Vector3d axis_;  //

    // 图像数据
    cv::Size img_size_;
    std::vector<std::string> image_paths_;
    std::vector<cv::Mat> images_;

    // 棋盘格参数
    cv::Size board_size_;
    float square_size_;  // 单格尺寸（米）
    int total_corners_;  // 总角点数量（width*height）

    // 标定数据
    int caliboard_points_size_;  // 单张标定板角点数量
    int points_size_;            // 标定点总数 points_size_=images_size*caliboard_points_size_

    std::vector<Eigen::Matrix<double, 1, 12>> B_;
};

#endif
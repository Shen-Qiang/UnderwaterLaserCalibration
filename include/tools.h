#ifndef TOOLS_H
#define TOOLS_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/SVD>

#include <Eigen/KroneckerProduct>
#include <Eigen/Polynomials>

#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include <dirent.h>
#include <algorithm>
#include <random>
#include <string>
#include <unordered_set>

using namespace Eigen;
using namespace std;

/**
 * @brief 计算将轴对准到（0，0，1）的旋转矩阵
 * @param axis 轴向量
 * @return 三维旋转矩阵
 * @note 经过校验
 */
Eigen::Matrix3d alignAxisToZ(const Eigen::Vector3d& axis);

Eigen::Vector3d pixelToCamRay(const cv::Point2f& pixel,
                              double fx,
                              double fy,
                              double cx,
                              double cy);

/**
 * @brief 折射光线计算函数（2D）
 * @param incident 入射光线
 * @param mu1 入射光线折射率
 * @param mu2 折射光线折射率
 * @param refracted 折射光线
 * @param tir 全反射标志
 * @note 注意空间法向量为（0，-1），经过校验
 */
void RefractedRay2D(const Eigen::Vector2d& incident,
                    double mu1,
                    double mu2,
                    Eigen::Vector2d& refracted,
                    int& tir);

/**
 * @brief 折射光线计算函数（3D）
 * @param incident 入射光线
 * @param normal 法向量
 * @param mu1 入射光线折射率
 * @param mu2 折射光线折射率
 * @param refracted 折射光线
 * @param tir 全反射标志
 * @note 经过校验
 */
void RefractedRay3D(const Eigen::Vector3d& incident,
                    const Eigen::Vector3d& normal,
                    double mu1,
                    double mu2,
                    Eigen::Vector3d& refracted,
                    int& tir);

// 求解12次多项式根
std::vector<std::complex<double>> solvePolynomialTwelve(const Eigen::VectorXd& coeffs);

/**
 * @brief 求解前向投影Case3（水+玻璃+空气），已知相机坐标系下的3D点，计算归一化像素坐标
 * @param d 第一介质分界面到相机距离（空气和玻璃分界面）
 * @param d2 第二介质分界面到相机距离（玻璃和水分界面）
 * @param n 法向量（0，0，-1）
 * @param mu 折射率（水+空气）
 * @param p 相机坐标系空间点3D坐标
 * @note 经过校验
 */
Eigen::Vector3d ForwardProjectionCase3(double d, double d2, const Eigen::Vector3d& n, const Eigen::Vector2d& mu, const Eigen::Vector3d& p);


/**
 * @brief 求解反向投影Case3（水+玻璃+空气），已知像素坐标，计算水界面出射点坐标和最终折射光线
 * @param d0 第一介质分界面到相机距离（空气和玻璃分界面）
 * @param d1 第二介质分界面到第一介质分界面距离（玻璃和水分界面）
 * @param n 法向量
 * @param mu 折射率（水+空气）
 * @param p 相机坐标系空间点3D坐标
 * @note 经过校验
 */
void backwardProjectionCase3(const double d0,
                            const double d1,
                            const Eigen::Vector3d axis, 
                            const Eigen::Vector3d mu, 
                            const Eigen::Vector3d v0_ray,
                            Eigen::Vector3d& p1,
                            Eigen::Vector3d& v2_ray);

void backwardProjectionCase3P0(const double d0,
                            const double d1,
                            const Eigen::Vector3d axis,
                            const Eigen::Vector3d mu,
                            const Eigen::Vector3d v0_ray,
                            Eigen::Vector3d& p0,
                            Eigen::Vector3d& p1,
                            Eigen::Vector3d& v2_ray);                            


/**
 * @brief Eigen::Matrix3d → 轴角向量
 * @param R 旋转矩阵
 * @return 三维轴角向量
 */
Eigen::Vector3d rotationMatrixToAngleAxis(const Eigen::Matrix3d& R);

/**
 * @brief 空间三维点计算（中点法）
 * @param left_center 左相机光线出射点
 * @param left_ray 左相机光线方向
 * @param right_center 右相机光线出射点
 * @param right_ray 右相机光线方向
 * @param mid_point 输出：三维重建点
 * @param reprojection_error 输出：重投影误差（两条光线的最短距离）
 * @note 经过校验
 */
void midPointReconstruction(const Eigen::Vector3d& left_center,
                            const Eigen::Vector3d& left_ray,
                            const Eigen::Vector3d& right_center,
                            const Eigen::Vector3d& right_ray,
                            Eigen::Vector3d& mid_point,
                            double* reprojection_error);

/**
 * @brief 判断两条光线是否满足对极约束
 * @param left_point 相机1的出射点
 * @param left_ray 相机1的光线方向（归一化）
 * @param right_point 相机2的出射点
 * @param right_ray 相机2的光线方向（归一化）
 * @return 
 */
double checkEpipolarConstraint(const Eigen::Vector3d& left_point,
                            const Eigen::Vector3d& left_ray,
                            const Eigen::Vector3d& right_point,
                            const Eigen::Vector3d& right_ray);

/**
 * @brief 计算离散外极曲线
 * @param left_point 相机1的出射点
 * @param left_ray 相机1的光线方向（归一化）
 * @param right_point 相机2的出射点
 * @param right_ray 相机2的光线方向（归一化）
 * @return 
 */
void calEpipolarPoint( Eigen::Vector3d water_point_right,
                         Eigen::Vector3d water_ray_right,
                        const cv::Mat& srcImg,
                        std::vector<Eigen::Vector2d> &result_points,
                        cv::Mat& imgline,
                        double fx,
                        double fy,
                        double cx,
                        double cy,
                        double mindistance,
                        double maxdistance,
                        double step_size,
                        double d0_left,
                        double d1_left,
                        Eigen::Vector3d n_left);



#endif
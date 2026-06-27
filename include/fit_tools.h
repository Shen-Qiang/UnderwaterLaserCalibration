// 曲线拟合工具函数
#ifndef FIT_TOOLS_H
#define FIT_TOOLS_H

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

struct ErrorPoint {
    double u;     // 控制参数
    double error; // 重投影误差
};

/**
 * @brief B样条曲线参数结构体
 * @details 存储定义一条B样条曲线所需的所有核心参数。
 *          拥有此结构体，即可在任何时候以任意精度重构曲线。
 */
struct BSplineCurve {
    int order;                      // 样条阶数 k (例如，三次B样条的 k=4)
    vector<Vector2d> control_points; // 控制点集合 P (大小为 n+1)
    vector<double> knot_vector;      // 节点矢量 U (大小为 n+1+k)
};


/**
 * B样条基函数递归计算 (De Boor 公式)
 * i: 基函数索引 (从0开始)
 * k: 阶数 (3次样条 k=4)
 * U: 节点矢量
 * u: 当前参数值
 */
double bspline_basis(int i, 
                    int k, 
                    const vector<double>& U, 
                    double u);

/**
 * 最小二乘非均匀三次B样条拟合 (使用 Eigen::Vector2d)
 * @param data_points 原始激光条纹中心线数据点 (m+1个)
 * @param num_control_points 待求解的控制点数量 (n+1个)
 * @param num_sample_points 最终光顺输出的离散点数量
 * @return 拟合后的平滑曲线点集
 */
vector<Eigen::Vector2d> fit_laser_stripe(const vector<Eigen::Vector2d>& data_points, 
                                    int num_control_points, 
                                    int num_sample_points,
                                    vector<double> &U_result);
    

/**
 * @brief 最小二乘非均匀三次B样条拟合，返回曲线核心参数
 * @param data_points 待拟合的原始数据点
 * @param num_control_points 用户期望的控制点数量 (n+1)
 * @return BSplineCurve 包含阶数、控制点和节点矢量的B样条曲线结构体
 */
BSplineCurve fit_bspline_curve(const vector<Vector2d>& data_points, 
                               int num_control_points);

/**
 * @brief 根据B样条参数计算曲线上任意一点的坐标
 * @param curve B样条曲线结构体
 * @param t 参数值，范围 [0.0, 1.0]
 * @return Vector2d 计算出的点坐标
 */
Vector2d calculate_bspline_point(const BSplineCurve& curve, 
                                double t);

// 简单的最小二乘线性拟合函数: y = kx + b
void linearFit(const std::vector<ErrorPoint>& points, 
                double& k, 
                double& b);


/**
 * @brief De Boor 算法：计算 B 样条在 u 处的点和导数
 * @param curve 曲线结构体
 * @param u 参数
 * @param point 返回坐标
 * @param grad 返回一阶导数 (dC/du)
 */
void evaluateBSpline(const BSplineCurve& curve, double u, Vector2d& point, Vector2d& grad);


/**
 * @brief 判断点 P 在曲线的侧向位置
 */
int getSide(const Vector2d& P, const BSplineCurve& curve);
                    
/**
 * @brief 精确计算交点
 */
Vector2d findIntersection(const Vector2d& P_left, 
    const Vector2d& P_right, 
    const BSplineCurve& curve);

#endif
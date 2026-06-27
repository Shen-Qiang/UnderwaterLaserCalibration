#ifndef LASER_EXTRACTOR_H
#define LASER_EXTRACTOR_H

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


/**
 * @brief 保存激光条纹边界点
 */
void writeBoundaryToTxt(const std::vector<Eigen::Vector2d>& boundary, 
                        const std::string& filename);

/**
 * @brief 基于LOG滤波器的激光条纹边界提取 (集成 ROI 裁剪与高级预处理)
 * 
 * @param srcImg           原始输入图像
 * @param rect             感兴趣区域 (ROI)
 * @param left_boundary    输出左边界点集
 * @param right_boundary   输出右边界点集
 * @param binaryThreshold  二值化阈值
 * @param logSigma         LOG滤波器标准差
 * @param logZeroThreshold 过零点梯度阈值
 * @return true 提取成功，false 失败
 */
bool extractLaserBoundariesLOG(const cv::Mat& srcImg,
                               const cv::Rect& rect,
                               std::vector<Eigen::Vector2d>& left_boundary,
                               std::vector<Eigen::Vector2d>& right_boundary,
                               int binaryThreshold,
                               double logSigma,
                               double logZeroThreshold) ;
/**
 * @brief 中轴变换提取激光条纹中心线
 */
std::vector<Eigen::Vector2d> medialAxisTransform(const std::vector<Eigen::Vector2d>& left_boundary,
                                                 const std::vector<Eigen::Vector2d>& right_boundary,
                                                 int max_iterations = 20,
                                                 double tolerance = 0.001);

// 暴露底层工具函数
cv::Mat createLOGFilter(double sigma);
std::vector<double> detectZeroCrossings(const double* signal, int length, double threshold);

/**
 * @brief 激光条纹中心线提取（LOG 边缘检测 + 中轴变换法）
 * 
 * @param srcImg            输入原始图像
 * @param rect              感兴趣区域 (ROI)
 * @param dstPoints         输出：提取到的中心线点集 (Eigen::Vector2d, 全局坐标)
 * @param imgline           输出：带有绘制结果的可视化图像 (BGR)
 * @param threshold         二值化阈值 (用于预处理 ROI 提取)
 * @param logSigma          LOG 滤波器标准差 (推荐 3.0)
 * @param logZeroThreshold  LOG 过零点梯度阈值 (推荐 0.01)
 * @param removeEdgePoints  移除两端点数 (减少端点噪声)
 */
void LogMedialCenterLineExtractionNew(const cv::Mat& srcImg, 
                                      const cv::Rect& rect,
                                      std::vector<Eigen::Vector2d>& dstPoints, 
                                      cv::Mat& imgline,
                                      int threshold = 50,
                                      double logSigma = 3.0,
                                      double logZeroThreshold = 0.01,
                                      int removeEdgePoints = 5);

#endif // LASER_STRIPE_EXTRACTOR_H

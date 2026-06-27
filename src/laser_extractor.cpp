#include "laser_extractor.h"
#include "tools.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <opencv2/imgproc.hpp>

// 保存激光条纹边界点
void writeBoundaryToTxt(const std::vector<Eigen::Vector2d>& boundary, 
    const std::string& filename) 
{
    // 1. 打开文件流 (ofstream: output file stream)
    std::ofstream outFile(filename);

    // 2. 检查文件是否打开成功
    if (!outFile.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    // 3. 设置输出精度（例如保留6位小数，可根据需要调整）
    outFile << std::fixed << std::setprecision(6);

    // 4. 遍历 vector，将每个点的 x 和 y 写入一行
    for (const auto& point : boundary) {
        // 每个点占据一行，中间用空格分隔
        outFile << point.x() << " " << point.y() << std::endl;
    }

    // 5. 关闭文件
    outFile.close();
    std::cout << "成功保存 " << boundary.size() << " 个点到 " << filename << std::endl;
}


// 2. LOG 滤波器核创建
cv::Mat createLOGFilter(double sigma) {
    int filter_size = 2 * static_cast<int>(std::ceil(3.0 * sigma)) + 1;
    int center = filter_size / 2;
    cv::Mat log_filter(filter_size, filter_size, CV_64FC1);
    double sigma2 = sigma * sigma;
    double sum = 0.0;
    for (int y = 0; y < filter_size; ++y) {
        for (int x = 0; x < filter_size; ++x) {
            double dx = static_cast<double>(x - center);
            double dy = static_cast<double>(y - center);
            double r2 = dx * dx + dy * dy;
            double val = -(1.0 / (CV_PI * sigma2 * sigma2)) * (1.0 - r2 / (2.0 * sigma2)) * std::exp(-r2 / (2.0 * sigma2));
            log_filter.at<double>(y, x) = val;
            sum += val;
        }
    }
    log_filter -= (sum / (filter_size * filter_size));
    return log_filter;
}

// 3. 亚像素过零点检测
std::vector<double> detectZeroCrossings(const double* signal, int length, double threshold) {
    std::vector<double> zero_crossings;
    for (int i = 0; i < length - 1; ++i) {
        if (signal[i] * signal[i + 1] < 0.0) {
            double grad = std::abs(signal[i + 1] - signal[i]);
            if (grad > threshold) {
                double pos = static_cast<double>(i) - signal[i] / (signal[i + 1] - signal[i]);
                zero_crossings.push_back(pos);
            }
        }
    }
    return zero_crossings;
}

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
                               double logZeroThreshold) 
{
    // ========== 1. 输入验证与 ROI 处理 ==========
    left_boundary.clear();
    right_boundary.clear();

    if (srcImg.empty()) {
        std::cerr << "[Error] Input image is empty!" << std::endl;
        return false;
    }

    // 检查并修正 ROI，确保其在图像范围内
    cv::Rect validRect = rect & cv::Rect(0, 0, srcImg.cols, srcImg.rows);
    if (validRect.area() == 0) {
        std::cerr << "[Error] ROI is outside image bounds!" << std::endl;
        return false;
    }

    // 提取 ROI 区域
    cv::Mat roiImg = srcImg(validRect);

    // ========== 2. 局部 ROI 预处理 ==========
    cv::Mat processedRoi;
    std::vector<cv::Vec4i> ranges;
    
    // 调用预处理函数（在 ROI 内部提取条纹区域）
    // 仿照灰度重心法：minArea=50, aspectRatio=1.0, expandPixels=10
    imgPreProcessNew(roiImg, binaryThreshold, processedRoi, ranges, 50.0, 1.0, 10);

    if (ranges.empty() || processedRoi.empty()) {
        std::cout << "[Warning] No valid regions found within the specified ROI!" << std::endl;
        return false;
    }

    // ========== 3. 准备 LOG 滤波器核 ==========
    cv::Mat logFilter = createLOGFilter(logSigma);

    // ========== 4. 遍历检测到的 Range 提取边界 ==========
    for (const auto& range : ranges) {
        int top    = range[0];
        int bottom = range[1];
        int left   = range[2];
        int right  = range[3];

        // 子 ROI 矩形
        cv::Rect subRoi(left, top, right - left, bottom - top);
        
        // 准备卷积数据
        cv::Mat regionData;
        processedRoi(subRoi).convertTo(regionData, CV_64FC1);

        // 应用 LOG 滤波
        cv::Mat logResponse;
        cv::filter2D(regionData, logResponse, CV_64FC1, logFilter, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

        // 逐行处理
        for (int r = 0; r < logResponse.rows; ++r) {
            int localY = top + r; // 相对于 validRect 的 Y 坐标
            
            const double* log_ptr = logResponse.ptr<double>(r);
            const uchar* intensity_ptr = processedRoi.ptr<uchar>(localY);

            // 亚像素过零点检测
            std::vector<double> zeros = detectZeroCrossings(log_ptr, logResponse.cols, logZeroThreshold);
            if (zeros.size() < 2) continue;

            // 寻找该行在 subRoi 范围内的最大亮度索引（用于区分左右边界）
            int max_idx_in_sub = 0;
            uchar max_val = 0;
            for (int c = left; c < right; ++c) {
                if (intensity_ptr[c] > max_val) {
                    max_val = intensity_ptr[c];
                    max_idx_in_sub = c - left;
                }
            }

            // 匹配靠近峰值的左右边界
            double best_l = -1.0, best_r = -1.0;
            for (int i = static_cast<int>(zeros.size()) - 1; i >= 0; --i) {
                if (zeros[i] < static_cast<double>(max_idx_in_sub)) { best_l = zeros[i]; break; }
            }
            for (size_t i = 0; i < zeros.size(); ++i) {
                if (zeros[i] > static_cast<double>(max_idx_in_sub)) { best_r = zeros[i]; break; }
            }

            // ========== 5. 坐标转换回全局图像坐标系 ==========
            if (best_l >= 0.0 && best_r >= 0.0) {
                // 全局 X = 过零点位置 + 子区域偏移 + ROI 偏移
                double globalX_L = best_l + left + validRect.x;
                double globalX_R = best_r + left + validRect.x;
                double globalY   = static_cast<double>(localY + validRect.y);

                left_boundary.emplace_back(globalX_L, globalY);
                right_boundary.emplace_back(globalX_R, globalY);
            }
        }
    }

    return !left_boundary.empty();
}


// 5. 中轴变换 (保持逻辑不变)
std::vector<Eigen::Vector2d> medialAxisTransform(const std::vector<Eigen::Vector2d>& left_boundary,
                                                 const std::vector<Eigen::Vector2d>& right_boundary,
                                                 int max_iterations, double tolerance) {
    std::vector<Eigen::Vector2d> centerline;
    if (left_boundary.empty() || right_boundary.empty()) return centerline;
    
    std::vector<Eigen::Vector2d> sl = left_boundary;
    std::vector<Eigen::Vector2d> sr = right_boundary;
    auto y_comp = [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) { return a.y() < b.y(); };
    std::sort(sl.begin(), sl.end(), y_comp);
    std::sort(sr.begin(), sr.end(), y_comp);
    
    for (const auto& p_left : sl) {
        auto it = std::min_element(sr.begin(), sr.end(), [&p_left](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
            return std::abs(a.y() - p_left.y()) < std::abs(b.y() - p_left.y());
        });
        Eigen::Vector2d p_right = *it;
        Eigen::Vector2d center = (p_left + p_right) / 2.0;
        double radius = (p_right - p_left).norm() / 2.0;

        for (int iter = 0; iter < max_iterations; ++iter) {
            auto closest_it = std::min_element(sr.begin(), sr.end(), [&center](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return (a - center).squaredNorm() < (b - center).squaredNorm();
            });
            Eigen::Vector2d p_right_new = *closest_it;
            Eigen::Vector2d center_new = (p_left + p_right_new) / 2.0;
            double radius_new = (p_right_new - p_left).norm() / 2.0;
            if (std::abs(radius_new - radius) < tolerance) { center = center_new; break; }
            center = center_new; radius = radius_new;
        }
        centerline.push_back(center);
    }
    return centerline;
}

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
                                      int threshold,
                                      double logSigma,
                                      double logZeroThreshold,
                                      int removeEdgePoints)
{
    // ========== 1. 输入验证与准备 ==========
    dstPoints.clear();
    if (srcImg.empty()) {
        std::cerr << "[Error] Input image is empty!" << std::endl;
        imgline = cv::Mat();
        return;
    }

    // 准备可视化图像
    cv::Mat imgRgb;
    if (srcImg.channels() == 1) {
        cv::cvtColor(srcImg, imgRgb, cv::COLOR_GRAY2BGR);
    } else {
        imgRgb = srcImg.clone();
    }

    // ========== 2. 调用 LOG 边界提取 ==========
    std::vector<Eigen::Vector2d> left_boundary;
    std::vector<Eigen::Vector2d> right_boundary;

    // 此处调用之前更新过的 extractLaserBoundariesLOG 函数
    bool success = extractLaserBoundariesLOG(srcImg, 
                                             rect, 
                                             left_boundary, 
                                             right_boundary, 
                                             threshold, 
                                             logSigma, 
                                             logZeroThreshold);

    if (!success || left_boundary.empty()) {
        std::cout << "[Warning] LOG method failed to extract boundaries in ROI!" << std::endl;
        imgline = imgRgb;
        return;
    }

    // --- 调用保存函数 ---
    writeBoundaryToTxt(left_boundary, "left_boundary.txt");
    writeBoundaryToTxt(right_boundary, "right_boundary.txt");


    // ========== 3. 执行中轴变换提取中心线 ==========
    // 调用之前实现的中轴变换函数
    std::vector<Eigen::Vector2d> centerline = medialAxisTransform(left_boundary, right_boundary);

    if (centerline.empty()) {
        std::cout << "[Warning] Medial Axis Transform returned no points!" << std::endl;
        imgline = imgRgb;
        return;
    }

    // ========== 4. 移除边缘点（减少端点噪声）==========
    if (centerline.size() > static_cast<std::size_t>(2 * removeEdgePoints)) {
        centerline.erase(centerline.begin(), centerline.begin() + removeEdgePoints);
        centerline.erase(centerline.end() - removeEdgePoints, centerline.end());
    }

    // ========== 5. 结果输出与可视化绘制 ==========
    dstPoints = centerline;

    // 在图上绘制中心线点 (绿色)
    for (const auto& pt : dstPoints) {
        cv::Point2f cv_pt(static_cast<float>(pt.x()), static_cast<float>(pt.y()));
        // 检查是否在图像范围内
        if (cv_pt.x >= 0 && cv_pt.x < imgRgb.cols && cv_pt.y >= 0 && cv_pt.y < imgRgb.rows) {
            cv::circle(imgRgb, cv_pt, 1, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
        }
    }

    // 可选：绘制检测到的左右边界 (淡蓝色和淡红色) 以便调试
    for (const auto& pt : left_boundary) 
        cv::circle(imgRgb, cv::Point2f(pt.x(), pt.y()), 0, cv::Scalar(0, 0, 150), -1);
    for (const auto& pt : right_boundary) 
        cv::circle(imgRgb, cv::Point2f(pt.x(), pt.y()), 0, cv::Scalar(150, 0, 0), -1);
    

    imgline = imgRgb;
    std::cout << "[Info] LOG+MedialAxis extraction finished. Points: " << dstPoints.size() << std::endl;
}

{
    // 1. 初始化与预处理
    dstPoints.clear();
    if (inputImage.empty()) return;
    imgLine = inputImage.clone();
    
    cv::Rect validRect = rect & cv::Rect(0, 0, inputImage.cols, inputImage.rows);
    cv::Mat proImg = inputImage(validRect);
    cv::Mat bwImg, grayImg;

    // 颜色空间处理
    if (proImg.channels() > 1) {
        std::vector<cv::Mat> channels;
        cv::split(proImg, channels);
        grayImg = channels[0]; // 提取蓝色通道 (根据原代码逻辑)
        cv::threshold(grayImg, bwImg, bwThr, 255, cv::THRESH_BINARY);
    } else {
        grayImg = proImg.clone();
        cv::threshold(grayImg, bwImg, bwThr, 255, cv::THRESH_BINARY);
    }

    // 形态学开运算：去除噪点
    cv::morphologyEx(bwImg, bwImg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    // 2. 提取区域 (轮廓查找)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bwImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Vec4i> Rois;
    for (const auto& cnt : contours) {
        if (cv::contourArea(cnt) < 50) continue;
        cv::Rect r = cv::boundingRect(cnt);
        // 扩展 10 像素以包含条纹边缘
        int top = std::max(0, r.y - 10);
        int bottom = std::min(bwImg.rows - 1, r.y + r.height + 10);
        int left = std::max(0, r.x - 10);
        int right = std::min(bwImg.cols - 1, r.x + r.width + 10);
        Rois.push_back(cv::Vec4i(top, bottom, left, right));
    }

    // 3. 求解高斯核 (Steger 理论核心：根据条纹宽度 w 计算 sigma = w/sqrt(3))
    double sigma = stripeWidth / std::sqrt(3.0);
    int range = static_cast<int>(std::round(3.0 * sigma));
    int gSize = 2 * range + 1;

    cv::Mat X(gSize, gSize, CV_64F), Y(gSize, gSize, CV_64F);
    for (int i = -range; i <= range; ++i) {
        for (int j = -range; j <= range; ++j) {
            X.at<double>(i + range, j + range) = static_cast<double>(j);
            Y.at<double>(i + range, j + range) = static_cast<double>(i);
        }
    }

    cv::Mat X2_Y2 = (X.mul(X) + Y.mul(Y)) / (2.0 * sigma * sigma);
    cv::Mat expTerm;
    cv::exp(-X2_Y2, expTerm);

    double coeff = 1.0 / (2.0 * CV_PI * std::pow(sigma, 4));
    cv::Mat DGaussx = coeff * (-X).mul(expTerm);
    cv::Mat DGaussy = DGaussx.t();
    cv::Mat DGaussxx = coeff * ((X.mul(X) / (sigma * sigma)) - 1.0).mul(expTerm);
    cv::Mat DGaussxy = (1.0 / (2.0 * CV_PI * std::pow(sigma, 6))) * (X.mul(Y)).mul(expTerm);
    cv::Mat DGaussyy = coeff * ((Y.mul(Y) / (sigma * sigma)) - 1.0).mul(expTerm);

    // 卷积核翻转 (用于 filter2D)
    cv::flip(DGaussx, DGaussx, -1);
    cv::flip(DGaussy, DGaussy, -1);
    cv::flip(DGaussxx, DGaussxx, -1);
    cv::flip(DGaussxy, DGaussxy, -1);
    cv::flip(DGaussyy, DGaussyy, -1);

    // 4. 计算各个 ROI 的 Hessian 矩阵
    cv::Mat visited = cv::Mat::zeros(grayImg.size(), CV_8UC1);
    std::vector<cv::Point2d> allLaserPoints;

    for (const auto& roi : Rois) {
        cv::Mat roiImgChunk = grayImg(cv::Range(roi[0], roi[1] + 1), cv::Range(roi[2], roi[3] + 1));
        roiImgChunk.convertTo(roiImgChunk, CV_64F);

        cv::Mat Dx, Dy, Dxx, Dxy, Dyy;
        cv::filter2D(roiImgChunk, Dx, CV_64F, DGaussx, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        cv::filter2D(roiImgChunk, Dy, CV_64F, DGaussy, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        cv::filter2D(roiImgChunk, Dxx, CV_64F, DGaussxx, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        cv::filter2D(roiImgChunk, Dxy, CV_64F, DGaussxy, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        cv::filter2D(roiImgChunk, Dyy, CV_64F, DGaussyy, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

        // 求解特征向量 (法线方向)
        cv::Mat tmp, v1x, v1y, mag;
        cv::sqrt((Dxx - Dyy).mul(Dxx - Dyy) + 4.0 * Dxy.mul(Dxy), tmp);
        v1x = 2.0 * Dxy;
        v1y = Dyy - Dxx + tmp;
        cv::magnitude(v1x, v1y, mag);
        v1x /= (mag + 1e-9);
        v1y /= (mag + 1e-9);

        // 特征值判定以选择最大响应方向
        cv::Mat Lambda1 = 0.5 * (Dxx + Dyy + tmp);
        cv::Mat Lambda2 = 0.5 * (Dxx + Dyy - tmp);
        cv::Mat check = cv::abs(Lambda1) < cv::abs(Lambda2);

        cv::Mat Ix_vec = v1x.clone();
        cv::Mat Iy_vec = v1y.clone();
        // 第二个特征向量垂直于第一个，通过原向量 (-y, x) 获取
        cv::Mat v2x = -v1y;
        cv::Mat v2y = v1x;
        v2x.copyTo(Ix_vec, check);
        v2y.copyTo(Iy_vec, check);

        // 亚像素偏移量 t = -(f' / f'')
        cv::Mat den = (Dxx.mul(Ix_vec.mul(Ix_vec)) + 2.0 * Dxy.mul(Ix_vec.mul(Iy_vec)) + Dyy.mul(Iy_vec.mul(Iy_vec)));
        cv::Mat t = -(Dx.mul(Ix_vec) + Dy.mul(Iy_vec)) / (den + 1e-9);
        cv::Mat px = t.mul(Ix_vec);
        cv::Mat py = t.mul(Iy_vec);

        std::vector<cv::Point2d> laserPoints;
        std::vector<cv::Point2i> candidatePoints;

        // 5. 提取候选亚像素点 (|t| <= 0.5)
        for (int y = 0; y < px.rows; ++y) {
            for (int x = 0; x < px.cols; ++x) {
                int gy = y + roi[0];
                int gx = x + roi[2];
                if (roiImgChunk.at<double>(y, x) <= sltThr || visited.at<uchar>(gy, gx) > 0 || bwImg.at<uchar>(gy, gx) == 0)
                    continue;

                if (std::abs(px.at<double>(y, x)) <= 0.5 && std::abs(py.at<double>(y, x)) <= 0.5) {
                    laserPoints.emplace_back(gx + px.at<double>(y, x), gy + py.at<double>(y, x));
                    candidatePoints.emplace_back(x, y);
                    visited.at<uchar>(gy, gx) = 255;
                }
            }
        }

        // 6. 筛选逻辑：防止同一行出现多个点（解决带状区域问题）
        if (filter && !laserPoints.empty()) {
            std::vector<cv::Point2d> filteredPoints;
            int n = static_cast<int>(laserPoints.size());
            std::vector<bool> processed(n, false);
            for (int i = 0; i < n; i++) {
                if (processed[i]) continue;
                processed[i] = true;

                int curY = candidatePoints[i].y;
                int minIndex = i;
                int num = 1;

                for (int j = i + 1; j < n; j++) {
                    if (candidatePoints[j].y != curY) break;
                    if (!processed[j]) {
                        num++;
                        processed[j] = true;
                    }
                }
                if (num >= 2) {
                    minIndex = i + (num - 1) / 2; // 取中间像素点作为中心
                }
                filteredPoints.push_back(laserPoints[minIndex]);
            }
            laserPoints = filteredPoints;
        }
        allLaserPoints.insert(allLaserPoints.end(), laserPoints.begin(), laserPoints.end());
    }

    // 7. 最终结果封装与可视化
    for (const auto& pt : allLaserPoints) {
        double finalX = pt.x + validRect.x;
        double finalY = pt.y + validRect.y;
        dstPoints.push_back(Eigen::Vector2d(finalX, finalY));
        cv::circle(imgLine, cv::Point2f(static_cast<float>(finalX), static_cast<float>(finalY)), 1, cv::Scalar(0, 0, 255), -1);
    }

    // 移除边缘点（根据需要可选）
    int removeNum = 10;
    if (dstPoints.size() > static_cast<size_t>(2 * removeNum)) {
        dstPoints.erase(dstPoints.begin(), dstPoints.begin() + removeNum);
        dstPoints.erase(dstPoints.end() - removeNum, dstPoints.end());
    }
}
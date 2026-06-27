// 曲线拟合工具函数
// fit_tools.cpp
#include "fit_tools.h"

/**
 * @brief B样条基函数递归计算 (De Boor 公式) - 修正版
 */
double bspline_basis(int i, 
    int k, 
    const vector<double>& U, 
    double u) 
{
    if (k == 1) {
        // 常规区间
        if (u >= U[i] && u < U[i + 1]) return 1.0;
        // 【修正1】：严格且安全地处理 u=1.0 的末端闭区间边界，避免浮点错误
        if (u == 1.0 && U[i] < 1.0 && U[i + 1] == 1.0) return 1.0;
        return 0.0;
    }

    double denom1 = U[i + k - 1] - U[i];
    double denom2 = U[i + k] - U[i + 1];

    double term1 = 0.0, term2 = 0.0;
    
    // 【修正2】：增加微小容差 > 1e-12 替代 != 0，防止浮点除 0 导致 NaN
    if (denom1 > 1e-12) {
        term1 = (u - U[i]) / denom1 * bspline_basis(i, k - 1, U, u);
    }
    if (denom2 > 1e-12) {
        term2 = (U[i + k] - u) / denom2 * bspline_basis(i + 1, k - 1, U, u);
    }

    return term1 + term2;
}

/**
 * @brief 最小二乘非均匀三次B样条拟合
 */
vector<Vector2d> fit_laser_stripe(const vector<Vector2d>& data_points, 
                                  int num_control_points, 
                                  int num_sample_points,
                                  vector<double> &U_result) 
{
    int m_plus_1 = data_points.size(); 
    int k = 4; // 三次B样条阶数

    if (num_control_points < k) {
        cerr << "错误: 控制点数量至少需要为 " << k << endl;
        return {};
    }

    // 1. 数据参数化：累计弦长法
    vector<double> u_bar(m_plus_1, 0.0);
    for (int i = 1; i < m_plus_1; ++i) {
        u_bar[i] = u_bar[i - 1] + (data_points[i] - data_points[i - 1]).norm();
    }
    double total_length = u_bar.back();
    if (total_length < 1e-9) return data_points; // 防止所有的点全在一个位置
    for (int i = 0; i < m_plus_1; ++i) u_bar[i] /= total_length; 

    // 2. 生成准均匀节点矢量 U
    vector<double> U(num_control_points + k, 0.0);
    
    // 尾部四重节点设为 1.0 (首部默认已初始化为0.0)
    for (int i = 0; i < k; ++i) U[num_control_points + k - 1 - i] = 1.0;
    
    // 【修正3核心】：对于逼近拟合，内部节点强制均匀分布！抛弃平均值法！
    int internal_knots = num_control_points - k;
    if (internal_knots > 0) {
        double step = 1.0 / (internal_knots + 1);
        for (int j = 1; j <= internal_knots; ++j) {
            U[j + k - 1] = j * step;
        }
    }

    // 3. 构建加权的最小二乘方程组 N * P = D
    MatrixXd N = MatrixXd::Zero(m_plus_1, num_control_points);
    MatrixXd D(m_plus_1, 2);

    // 【修正4】：端点高权重锚定。给予首尾极高权重，强制曲线紧贴起始点和终止点
    double weight_end = 1000.0; 

    for (int i = 0; i < m_plus_1; ++i) {
        double w = (i == 0 || i == m_plus_1 - 1) ? weight_end : 1.0;
        
        D.row(i) = data_points[i].transpose() * w; 
        for (int j = 0; j < num_control_points; ++j) {
            N(i, j) = bspline_basis(j, k, U, u_bar[i]) * w;
        }
    }

    // 4. 求解控制点矩阵 P (n+1 x 2)
    // 【修正5】：引入 Ridge Regression(岭回归) 1e-6，防止节点区间内无数据点导致的矩阵病态报错
    MatrixXd A = N.transpose() * N + 1e-6 * MatrixXd::Identity(num_control_points, num_control_points);
    MatrixXd B = N.transpose() * D;
    MatrixXd P = A.ldlt().solve(B);

    // 5. 均匀重采样生成光顺曲线
    vector<Vector2d> fitted_curve;
    fitted_curve.reserve(num_sample_points);

    for (int i = 0; i < num_sample_points; ++i) {
        double t = static_cast<double>(i) / (num_sample_points - 1);
        if (t > 1.0) t = 1.0; // 强制钳位
        
        Vector2d pt_fit = Vector2d::Zero();
        for (int j = 0; j < num_control_points; ++j) {
            double basis_val = bspline_basis(j, k, U, t);
            pt_fit += basis_val * P.row(j).transpose();
        }
        fitted_curve.push_back(pt_fit);
    }
    U_result = U;

    return fitted_curve;
}

/**
 * @brief 最小二乘非均匀三次B样条拟合，返回曲线核心参数
 * @param data_points 待拟合的原始数据点
 * @param num_control_points 用户期望的控制点数量 (n+1)
 * @return BSplineCurve 包含阶数、控制点和节点矢量的B样条曲线结构体
 */
BSplineCurve fit_bspline_curve(const vector<Vector2d>& data_points, 
                               int num_control_points) 
{
    int m_plus_1 = data_points.size(); 
    int k = 4; // 三次B样条阶数

    if (num_control_points < k || m_plus_1 < k) {
        cerr << "错误: 控制点和数据点数量至少需要为 " << k << endl;
        return {}; // 返回一个空的结构体表示失败
    }

    // 1. 数据参数化：累计弦长法
    vector<double> u_bar(m_plus_1, 0.0);
    for (int i = 1; i < m_plus_1; ++i) {
        u_bar[i] = u_bar[i - 1] + (data_points[i] - data_points[i - 1]).norm();
    }
    double total_length = u_bar.back();
    if (total_length < 1e-9) {
        cerr << "警告: 所有数据点重合，无法进行参数化。" << endl;
        return {};
    }
    for (int i = 0; i < m_plus_1; ++i) u_bar[i] /= total_length; 

    // 2. 生成准均匀节点矢量 U
    vector<double> U(num_control_points + k, 0.0);
    for (int i = 0; i < k; ++i) U[num_control_points + k - 1 - i] = 1.0;
    int internal_knots_num = num_control_points - k;
    if (internal_knots_num > 0) {
        double step = 1.0 / (internal_knots_num + 1);
        for (int j = 1; j <= internal_knots_num; ++j) {
            U[j + k - 1] = j * step;
        }
    }

    // 3. 构建加权的最小二乘方程组 N * P = D
    MatrixXd N = MatrixXd::Zero(m_plus_1, num_control_points);
    MatrixXd D(m_plus_1, 2);
    double weight_end = 1000.0; 

    for (int i = 0; i < m_plus_1; ++i) {
        double w = (i == 0 || i == m_plus_1 - 1) ? weight_end : 1.0;
        D.row(i) = data_points[i].transpose() * w; 
        // 为了使最后一个点u=1.0能被计算，修正基函数调用
        double u_param = u_bar[i];
        if (u_param >= 1.0) u_param = 1.0 - 1e-9; // 钳位到有效区间内
        for (int j = 0; j < num_control_points; ++j) {
            N(i, j) = bspline_basis(j, k, U, u_param) * w;
        }
    }

    // 4. 求解控制点矩阵 P
    MatrixXd A = N.transpose() * N + 1e-6 * MatrixXd::Identity(num_control_points, num_control_points);
    MatrixXd B = N.transpose() * D;
    MatrixXd P_mat = A.ldlt().solve(B);

    // 将 Eigen::MatrixXd 转换为 vector<Vector2d>
    vector<Vector2d> control_points_vec;
    control_points_vec.reserve(num_control_points);
    for (int i = 0; i < P_mat.rows(); ++i) {
        control_points_vec.emplace_back(P_mat(i, 0), P_mat(i, 1));
    }

    // 5. 封装并返回结果
    BSplineCurve result_curve;
    result_curve.order = k;
    result_curve.control_points = control_points_vec;
    result_curve.knot_vector = U;

    return result_curve;
}

/**
 * @brief 根据B样条参数计算曲线上任意一点的坐标
 * @param curve B样条曲线结构体
 * @param t 参数值，范围 [0.0, 1.0]
 * @return Vector2d 计算出的点坐标
 */
Vector2d calculate_bspline_point(const BSplineCurve& curve, 
                                double t) 
{
    if (curve.control_points.empty() || curve.knot_vector.empty()) {
        cerr << "错误: 提供的B样条曲线参数无效。" << endl;
        return Vector2d::Zero();
    }
    
    // 将参数t钳位到B样条定义的有效域内
    if (t < 0.0) t = 0.0;
    if (t >= 1.0) t = 1.0 - 1e-9; // 遵循基函数定义，t=1.0时可能无定义

    Vector2d point = Vector2d::Zero();
    for (size_t i = 0; i < curve.control_points.size(); ++i) {
        double basis_val = bspline_basis(i, curve.order, curve.knot_vector, t);
        point += basis_val * curve.control_points[i];
    }
    return point;
}

// 简单的最小二乘线性拟合函数: y = kx + b
void linearFit(const std::vector<ErrorPoint>& points, 
                double& k, 
                double& b) 
{
    double n = static_cast<double>(points.size());
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (const auto& p : points) {
        sumX += p.u;
        sumY += p.error;
        sumXY += p.u * p.error;
        sumX2 += p.u * p.u;
    }
    k = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    b = (sumY - k * sumX) / n;
}

/**
 * @brief De Boor 算法：计算 B 样条在 u 处的点和导数
 * @param curve 曲线结构体
 * @param u 参数
 * @param point 返回坐标
 * @param grad 返回一阶导数 (dC/du)
 */
void evaluateBSpline(const BSplineCurve& curve, double u, Vector2d& point, Vector2d& grad) {
    int k = curve.order;
    int n = curve.control_points.size() - 1;
    const vector<double>& U = curve.knot_vector;

    // 1. 找到 u 所在的节点区间 [U_p, U_{p+1})
    int p = -1;
    for (int i = k - 1; i <= n; ++i) {
        if (u >= U[i] && u <= U[i + 1]) {
            p = i;
            break;
        }
    }
    if (p == -1) {
        if (u < U[0]) p = k - 1;
        else p = n;
    }

    // 2. 计算点坐标 (De Boor 递归)
    vector<Vector2d> d(k);
    for (int j = 0; j < k; ++j) d[j] = curve.control_points[p - k + 1 + j];

    for (int r = 1; r < k; ++r) {
        for (int j = k - 1; j >= r; --j) {
            int idx = p - k + 1 + j;
            double alpha = (u - U[idx]) / (U[idx + k - r] - U[idx]);
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
        }
    }
    point = d[k - 1];

    // 3. 计算一阶导数 (B样条导数性质)
    vector<Vector2d> d_grad(k);
    for (int j = 0; j < k; ++j) d_grad[j] = curve.control_points[p - k + 1 + j];
    
    // 导数是 k-1 阶的样条
    for (int r = 1; r < k - 1; ++r) {
        for (int j = k - 1; j >= r; --j) {
            int idx = p - k + 1 + j;
            double alpha = (u - U[idx]) / (U[idx + k - 1 - r] - U[idx]);
            d_grad[j] = (1.0 - alpha) * d_grad[j - 1] + alpha * d_grad[j];
        }
    }
    // 最终导数公式
    double pre1 = (k - 1) / (U[p + 1] - U[p - k + 2]); // 简化处理
    grad = (k - 1) * (d_grad[k - 1] - d_grad[k - 2]) / (U[p + 1] - U[p - k + 2]);
}

/**
 * @brief 判断点 P 在曲线的侧向位置
 */
int getSide(const Vector2d& P, const BSplineCurve& curve) {
    // 粗略映射：假设竖向曲线 u 与 y 线性相关
    double u_start = curve.knot_vector.front();
    double u_end = curve.knot_vector.back();
    
    // 简单搜索：找到 y 最接近的点对应的 u
    double best_u = u_start;
    double min_dy = 1e9;
    for (int i = 0; i <= 20; ++i) {
        double u = u_start + (u_end - u_start) * i / 20.0;
        Vector2d pt, gd;
        evaluateBSpline(curve, u, pt, gd);
        if (abs(pt.y() - P.y()) < min_dy) {
            min_dy = abs(pt.y() - P.y());
            best_u = u;
        }
    }

    Vector2d p_on_curve, grad;
    evaluateBSpline(curve, best_u, p_on_curve, grad);
    return (P.x() > p_on_curve.x()) ? 1 : -1;
}

/**
 * @brief 精确计算交点
 */
Vector2d findIntersection(const Vector2d& P_left, 
    const Vector2d& P_right, 
    const BSplineCurve& curve) 
{
    // 1. 利用 B 样条的凸包特性或简单采样，极速定位目标节点区间
    // 假设曲线主要是竖向的，我们直接寻找 Y 落在目标范围内的节点区间
    double target_y = (P_left.y() + P_right.y()) * 0.5;
    int n = curve.control_points.size() - 1;
    int k = curve.order;
    
    // 寻找起始参数 u 的初值：只检查节点处的 Y 值（计算量极小）
    double u_start = curve.knot_vector[k - 1];
    double u_end = curve.knot_vector[n + 1];
    
    // 快速扫描节点矢量，缩小 u 的范围（不再全曲线采样 100 点）
    double u_low = u_start, u_high = u_end;
    for (int i = k - 1; i <= n; ++i) {
        double u_knot = curve.knot_vector[i];
        Vector2d pt, gd;
        evaluateBSpline(curve, u_knot, pt, gd); // 只在节点处采样
        if (pt.y() < target_y) u_low = u_knot;
        else { u_high = u_knot; break; }
    }

    // 2. 割线法 (Secant Method) 迭代
    // 相比二分法，它收敛极快；相比牛顿法，它更稳定且不依赖导数
    auto getError = [&](double u_test) {
        Vector2d C, gd;
        evaluateBSpline(curve, u_test, C, gd);
        double t = (C.y() - P_left.y()) / (P_right.y() - P_left.y());
        double lineX = P_left.x() + t * (P_right.x() - P_left.x());
        return C.x() - lineX;
    };

    double u0 = u_low;
    double u1 = u_high;
    double f0 = getError(u0);
    double f1 = getError(u1);
    double u_res = u1;

    // 割线法通常只需 4-6 次迭代
    for (int i = 0; i < 6; ++i) {
        if (std::abs(f1 - f0) < 1e-12) break; // 防止除零
        
        u_res = u1 - f1 * (u1 - u0) / (f1 - f0);
        
        // 投影回合法区间，保证稳定性
        u_res = std::max(u_low, std::min(u_high, u_res));
        
        if (std::abs(u_res - u1) < 1e-8) break; // 参数收敛

        u0 = u1; f0 = f1;
        u1 = u_res; f1 = getError(u1);
        
        if (std::abs(f1) < 1e-7) break; // 误差收敛
    }

    Vector2d final_p, final_gd;
    evaluateBSpline(curve, u_res, final_p, final_gd);
    return final_p;

}

// multimedium_calibrator.cpp
#include "multimedium_calibrator.h"

// 随机抽取n个样本
std::vector<Eigen::Matrix<double, 1, 12>> RandomSampleRANSAC_Shuffle(
    const std::vector<Eigen::Matrix<double, 1, 12>>& B_,
    int n,
    std::vector<int>& ids) 
{
    std::vector<Eigen::Matrix<double, 1, 12>> samples;
    ids.clear();

    if (B_.empty() || n <= 0) {
        return samples;
    }

    int total_samples = static_cast<int>(B_.size());

    // 如果n大于等于总样本数，返回所有样本
    if (n >= total_samples) {
        return B_;
    }

    // 生成所有索引
    std::vector<int> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);

    // 随机打乱索引
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    // 取前n个索引对应的样本
    // indices.clear();
    // indices = {70, 52, 23, 7, 50, 29, 44, 38};
    samples.reserve(n);
    for (int i = 0; i < n; i++) {
        samples.push_back(B_[indices[i]]);  // 随机采样
        // samples.push_back(B_[i+10]);  // 前8个
        ids.push_back(indices[i]);
    }

    return samples;
}

// 直接求解二次方程：a*x^2 + b*x + c = 0
Eigen::VectorXcd solveQuadraticDirect(double a, double b, double c) 
{
    Eigen::VectorXd coeffs(3);
    coeffs << c, b, a;  // Eigen多项式系数从常数项开始

    Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
    solver.compute(coeffs);

    return solver.roots();
}

// 从E32恢复E33(先求解z)
std::vector<Eigen::Matrix<double, 3, 3>> MultimediumCalibrator::SolveLastColOfEmatrix(const Eigen::Matrix<double, 3, 2>& x) 
{
    // ========== Step 1: 归一化（对应MATLAB的fac计算） ==========
    Eigen::VectorXd tt = x.array().abs().reshaped<Eigen::ColMajor>();  // 展平为6×1列向量
    double fac = 1.0 / tt.mean();
    Eigen::Matrix<double, 3, 2> x_norm = x * fac;

    // 提取前两列元素（对应e1~e6）
    double e1 = x_norm(0, 0), e2 = x_norm(1, 0), e3 = x_norm(2, 0);
    double e4 = x_norm(0, 1), e5 = x_norm(1, 1), e6 = x_norm(2, 1);

    // ========== Step 2: 构造多项式系数s1, s2, s3（完全对应MATLAB表达式） ==========
    // 先定义MATLAB中重复出现的子表达式，减少冗余计算
    // 子表达式1: A = e1^3*e5^2 - 2*e1^2*e2*e4*e5 + ...
    double A = pow(e1, 3) * pow(e5, 2) - 2 * pow(e1, 2) * e2 * e4 * e5 + e1 * pow(e2, 2) * pow(e4, 2) - e1 * pow(e2, 2) * pow(e6, 2) + e1 * pow(e3, 2) * pow(e5, 2) + 2 * pow(e2, 2) * e3 * e4 * e6 - 2 * e2 * pow(e3, 2) * e4 * e5;
    // 子表达式2: B = -e1^2*e5*e6^2 + 2*e1*e2*e4*e6^2 + ...
    double B = -pow(e1, 2) * e5 * pow(e6, 2) + 2 * e1 * e2 * e4 * pow(e6, 2) + pow(e2, 2) * e5 * pow(e6, 2) - 2 * e2 * e3 * pow(e4, 2) * e6 - 2 * e2 * e3 * pow(e5, 2) * e6 + pow(e3, 2) * pow(e4, 2) * e5 + pow(e3, 2) * pow(e5, 3);
    // 子表达式3: C = e1^2*e5^3 - 2*e1*e2*e4*e5^2 + ...
    double C = pow(e1, 2) * pow(e5, 3) - 2 * e1 * e2 * e4 * pow(e5, 2) + pow(e2, 2) * pow(e4, 2) * e5 + pow(e2, 2) * e5 * pow(e6, 2) - 2 * e2 * e3 * pow(e5, 2) * e6 + pow(e3, 2) * pow(e5, 3);
    // 子表达式4: D = e1^3*e6^2 - 2*e1^2*e3*e4*e6 + ...
    double D = pow(e1, 3) * pow(e6, 2) - 2 * pow(e1, 2) * e3 * e4 * e6 + e1 * pow(e2, 2) * pow(e6, 2) + e1 * pow(e3, 2) * pow(e4, 2) - e1 * pow(e3, 2) * pow(e5, 2) - 2 * pow(e2, 2) * e3 * e4 * e6 + 2 * e2 * pow(e3, 2) * e4 * e5;
    // 子表达式5: E_sub = 2*e1^3*e5*e6 - 2*e1^2*e2*e4*e6 - ...
    double E_sub = 2 * pow(e1, 3) * e5 * e6 - 2 * pow(e1, 2) * e2 * e4 * e6 - 2 * pow(e1, 2) * e3 * e4 * e5 + 2 * e1 * pow(e2, 2) * e5 * e6 + 2 * e1 * e2 * e3 * pow(e4, 2) - 2 * e1 * e2 * e3 * pow(e5, 2) - 2 * e1 * e2 * e3 * pow(e6, 2) + 2 * e1 * pow(e3, 2) * e5 * e6 - 2 * pow(e2, 3) * e4 * e6 + 2 * pow(e2, 2) * e3 * e4 * e5 + 2 * e2 * pow(e3, 2) * e4 * e6 - 2 * pow(e3, 3) * e4 * e5;
    // 子表达式6: F = 2*e1^2*e5^2*e6 - 4*e1*e2*e4*e5*e6 + ...
    double F = 2 * pow(e1, 2) * pow(e5, 2) * e6 - 4 * e1 * e2 * e4 * e5 * e6 + 2 * pow(e2, 2) * pow(e4, 2) * e6 + 2 * pow(e2, 2) * pow(e6, 3) - 4 * e2 * e3 * e5 * pow(e6, 2) + 2 * pow(e3, 2) * pow(e5, 2) * e6;
    // 子表达式7: G = e1^3*e2^2*e6^2 - 2*e1^3*e2*e3*e5*e6 + ...（超长表达式，完整对应MATLAB）
    double G = pow(e1, 3) * pow(e2, 2) * pow(e6, 2) - 2 * pow(e1, 3) * e2 * e3 * e5 * e6 + pow(e1, 3) * pow(e3, 2) * pow(e5, 2) + e1 * pow(e2, 4) * pow(e6, 2) - 2 * e1 * pow(e2, 3) * e3 * e5 * e6 + e1 * pow(e2, 2) * pow(e3, 2) * pow(e5, 2) + e1 * pow(e2, 2) * pow(e3, 2) * pow(e6, 2) + e1 * pow(e2, 2) * pow(e4, 2) * pow(e6, 2) - e1 * pow(e2, 2) * pow(e5, 2) * pow(e6, 2) - e1 * pow(e2, 2) * pow(e6, 4) - 2 * e1 * e2 * pow(e3, 3) * e5 * e6 - 2 * e1 * e2 * e3 * pow(e4, 2) * e5 * e6 + 2 * e1 * e2 * e3 * pow(e5, 3) * e6 + 2 * e1 * e2 * e3 * e5 * pow(e6, 3) + e1 * pow(e3, 4) * pow(e5, 2) + e1 * pow(e3, 2) * pow(e4, 2) * pow(e5, 2) - e1 * pow(e3, 2) * pow(e5, 4) - e1 * pow(e3, 2) * pow(e5, 2) * pow(e6, 2) + 2 * pow(e2, 3) * e4 * e5 * pow(e6, 2) - 4 * pow(e2, 2) * e3 * e4 * pow(e5, 2) * e6 + 2 * pow(e2, 2) * e3 * e4 * pow(e6, 3) + 2 * e2 * pow(e3, 2) * e4 * pow(e5, 3) - 4 * e2 * pow(e3, 2) * e4 * e5 * pow(e6, 2) + 2 * pow(e3, 3) * e4 * pow(e5, 2) * e6;
    // 子表达式8: H = -e1^2*e2^2*e5*e6^2 + 2*e1^2*e2*e3*e5^2*e6 - ...（完整对应MATLAB）
    double H = -pow(e1, 2) * pow(e2, 2) * e5 * pow(e6, 2) + 2 * pow(e1, 2) * e2 * e3 * pow(e5, 2) * e6 - pow(e1, 2) * pow(e3, 2) * pow(e5, 3) + 2 * e1 * pow(e2, 3) * e4 * pow(e6, 2) - 4 * e1 * pow(e2, 2) * e3 * e4 * e5 * e6 + 2 * e1 * e2 * pow(e3, 2) * e4 * pow(e5, 2) + pow(e2, 4) * e5 * pow(e6, 2) - 2 * pow(e2, 3) * e3 * pow(e5, 2) * e6 + 2 * pow(e2, 3) * e3 * pow(e6, 3) + pow(e2, 2) * pow(e3, 2) * pow(e5, 3) - 5 * pow(e2, 2) * pow(e3, 2) * e5 * pow(e6, 2) + pow(e2, 2) * pow(e4, 2) * e5 * pow(e6, 2) + pow(e2, 2) * pow(e5, 3) * pow(e6, 2) + pow(e2, 2) * e5 * pow(e6, 4) + 4 * e2 * pow(e3, 3) * pow(e5, 2) * e6 - 2 * e2 * e3 * pow(e4, 2) * pow(e5, 2) * e6 - 2 * e2 * e3 * pow(e5, 4) * e6 - 2 * e2 * e3 * pow(e5, 2) * pow(e6, 3) - pow(e3, 4) * pow(e5, 3) + pow(e3, 2) * pow(e4, 2) * pow(e5, 3) + pow(e3, 2) * pow(e5, 5) + pow(e3, 2) * pow(e5, 3) * pow(e6, 2);
    // 子表达式9: I = 2*e1^3*e6^2 - 4*e1^2*e3*e4*e6 + ...
    double I = 2 * pow(e1, 3) * pow(e6, 2) - 4 * pow(e1, 2) * e3 * e4 * e6 + 2 * e1 * pow(e2, 2) * pow(e6, 2) + 2 * e1 * pow(e3, 2) * pow(e4, 2) - 2 * e1 * pow(e3, 2) * pow(e5, 2) - 4 * pow(e2, 2) * e3 * e4 * e6 + 4 * e2 * pow(e3, 2) * e4 * e5;

    // 计算s1（对应MATLAB的s1）
    double s1 = pow(A * B, 2) + pow(C * D, 2) - C * B * pow(E_sub, 2) + pow(F, 2) * A * D + C * A * I * B + F * A * B * E_sub - C * F * D * E_sub;

    // 计算s2（对应MATLAB的s2）
    double s2 = 2 * A * pow(B, 2) * G + B * pow(E_sub, 2) * H - 2 * C * pow(D, 2) * H - (A * H - C * G) * I * B + pow(F, 2) * D * G + F * B * E_sub * G + F * D * E_sub * H;

    // 计算s3（对应MATLAB的s3）
    double s3 = pow(B * G, 2) + pow(D * H, 2) - I * B * H * G;

    Eigen::VectorXcd roots = solveQuadraticDirect(s1, s2, s3);
    std::cout << "多项式根:" << std::endl;
    for (int i = 0; i < roots.size(); i++) {
        std::cout << "根 " << i << ": " << roots(i).real() << " + " << roots(i).imag() << "i" << std::endl;
    }

    // 筛选正实数根（对应MATLAB的find(ss>0)）
    std::vector<double> valid_roots;
    for (int i = 0; i < roots.size(); ++i) {
        // 检查是否为实数（虚部趋近于0）且大于0
        if (std::abs(roots(i).imag()) < 1e-8 && roots(i).real() > 1e-8) {
            valid_roots.push_back(roots(i).real());
        }
    }

    if (valid_roots.empty()) {
        return {};  // 无有效解
    }

    // 构造cSol = [sqrt(ss); -sqrt(ss)]（对应MATLAB的符号相反解）
    std::vector<double> cSol;
    for (double r : valid_roots) {
        double sqrt_r = std::sqrt(r);
        cSol.push_back(sqrt_r);
        cSol.push_back(-sqrt_r);
    }

    // ========== Step 4: 还原E矩阵第三列（对应MATLAB的for循环） ==========
    std::vector<Eigen::Matrix<double, 3, 3>> Ef_list;
    for (double c : cSol) {
        // 计算x1, x2, x3, y1, y2, y3（对应MATLAB中的同名变量）
        // ========== 严格复刻原公式：计算 x1/x2/x3 ==========
        double x1 = pow(e1, 3) * pow(e6, 2) - 2 * pow(e1, 2) * e3 * e4 * e6 + e1 * pow(e2, 2) * pow(e6, 2) + e1 * pow(e3, 2) * pow(e4, 2) - e1 * pow(e3, 2) * pow(e5, 2) - 2 * pow(e2, 2) * e3 * e4 * e6 + 2 * e2 * pow(e3, 2) * e4 * e5;

        double x2 = -2 * c * pow(e1, 3) * e5 * e6 + 2 * c * pow(e1, 2) * e2 * e4 * e6 + 2 * c * pow(e1, 2) * e3 * e4 * e5 - 2 * c * e1 * pow(e2, 2) * e5 * e6 - 2 * c * e1 * e2 * e3 * pow(e4, 2) + 2 * c * e1 * e2 * e3 * pow(e5, 2) + 2 * c * e1 * e2 * e3 * pow(e6, 2) - 2 * c * e1 * pow(e3, 2) * e5 * e6 + 2 * c * pow(e2, 3) * e4 * e6 - 2 * c * pow(e2, 2) * e3 * e4 * e5 - 2 * c * e2 * pow(e3, 2) * e4 * e6 + 2 * c * pow(e3, 3) * e4 * e5;

        double x3 = pow(c, 2) * pow(e1, 3) * pow(e5, 2) - 2 * pow(c, 2) * pow(e1, 2) * e2 * e4 * e5 + pow(c, 2) * e1 * pow(e2, 2) * pow(e4, 2) - pow(c, 2) * e1 * pow(e2, 2) * pow(e6, 2) + pow(c, 2) * e1 * pow(e3, 2) * pow(e5, 2) + 2 * pow(c, 2) * pow(e2, 2) * e3 * e4 * e6 - 2 * pow(c, 2) * e2 * pow(e3, 2) * e4 * e5 + pow(e1, 3) * pow(e2, 2) * pow(e6, 2) - 2 * pow(e1, 3) * e2 * e3 * e5 * e6 + pow(e1, 3) * pow(e3, 2) * pow(e5, 2) + e1 * pow(e2, 4) * pow(e6, 2) - 2 * e1 * pow(e2, 3) * e3 * e5 * e6 + e1 * pow(e2, 2) * pow(e3, 2) * pow(e5, 2) + e1 * pow(e2, 2) * pow(e3, 2) * pow(e6, 2) + e1 * pow(e2, 2) * pow(e4, 2) * pow(e6, 2) - e1 * pow(e2, 2) * pow(e5, 2) * pow(e6, 2) - e1 * pow(e2, 2) * pow(e6, 4) - 2 * e1 * e2 * pow(e3, 3) * e5 * e6 - 2 * e1 * e2 * e3 * pow(e4, 2) * e5 * e6 + 2 * e1 * e2 * e3 * pow(e5, 3) * e6 + 2 * e1 * e2 * e3 * e5 * pow(e6, 3) + e1 * pow(e3, 4) * pow(e5, 2) + e1 * pow(e3, 2) * pow(e4, 2) * pow(e5, 2) - e1 * pow(e3, 2) * pow(e5, 4) - e1 * pow(e3, 2) * pow(e5, 2) * pow(e6, 2) + 2 * pow(e2, 3) * e4 * e5 * pow(e6, 2) - 4 * pow(e2, 2) * e3 * e4 * pow(e5, 2) * e6 + 2 * pow(e2, 2) * e3 * e4 * pow(e6, 3) + 2 * e2 * pow(e3, 2) * e4 * pow(e5, 3) - 4 * e2 * pow(e3, 2) * e4 * e5 * pow(e6, 2) + 2 * pow(e3, 3) * e4 * pow(e5, 2) * e6;

        // ========== 严格复刻原公式：计算 y1/y2/y3 ==========
        double y1 = -pow(e1, 2) * e5 * pow(e6, 2) + 2 * e1 * e2 * e4 * pow(e6, 2) + pow(e2, 2) * e5 * pow(e6, 2) - 2 * e2 * e3 * pow(e4, 2) * e6 - 2 * e2 * e3 * pow(e5, 2) * e6 + pow(e3, 2) * pow(e4, 2) * e5 + pow(e3, 2) * pow(e5, 3);

        double y2 = 2 * c * pow(e1, 2) * pow(e5, 2) * e6 - 4 * c * e1 * e2 * e4 * e5 * e6 + 2 * c * pow(e2, 2) * pow(e4, 2) * e6 + 2 * c * pow(e2, 2) * pow(e6, 3) - 4 * c * e2 * e3 * e5 * pow(e6, 2) + 2 * c * pow(e3, 2) * pow(e5, 2) * e6;

        double y3 = -pow(c, 2) * pow(e1, 2) * pow(e5, 3) + 2 * pow(c, 2) * e1 * e2 * e4 * pow(e5, 2) - pow(c, 2) * pow(e2, 2) * pow(e4, 2) * e5 - pow(c, 2) * pow(e2, 2) * e5 * pow(e6, 2) + 2 * pow(c, 2) * e2 * e3 * pow(e5, 2) * e6 - pow(c, 2) * pow(e3, 2) * pow(e5, 3) - pow(e1, 2) * pow(e2, 2) * e5 * pow(e6, 2) + 2 * pow(e1, 2) * e2 * e3 * pow(e5, 2) * e6 - pow(e1, 2) * pow(e3, 2) * pow(e5, 3) + 2 * e1 * pow(e2, 3) * e4 * pow(e6, 2) - 4 * e1 * pow(e2, 2) * e3 * e4 * e5 * e6 + 2 * e1 * e2 * pow(e3, 2) * e4 * pow(e5, 2) + pow(e2, 4) * e5 * pow(e6, 2) - 2 * pow(e2, 3) * e3 * pow(e5, 2) * e6 + 2 * pow(e2, 3) * e3 * pow(e6, 3) + pow(e2, 2) * pow(e3, 2) * pow(e5, 3) - 5 * pow(e2, 2) * pow(e3, 2) * e5 * pow(e6, 2) + pow(e2, 2) * pow(e4, 2) * e5 * pow(e6, 2) + pow(e2, 2) * pow(e5, 3) * pow(e6, 2) + pow(e2, 2) * e5 * pow(e6, 4) + 4 * e2 * pow(e3, 3) * pow(e5, 2) * e6 - 2 * e2 * e3 * pow(e4, 2) * pow(e5, 2) * e6 - 2 * e2 * e3 * pow(e5, 4) * e6 - 2 * e2 * e3 * pow(e5, 2) * pow(e6, 3) - pow(e3, 4) * pow(e5, 3) + pow(e3, 2) * pow(e4, 2) * pow(e5, 3) + pow(e3, 2) * pow(e5, 5) + pow(e3, 2) * pow(e5, 3) * pow(e6, 2);

        // 计算b = -(x1*y3 - x3*y1)/(x1*y2 - x2*y1)
        double denominator = x1 * y2 - x2 * y1;
        if (std::abs(denominator) < 1e-12) {
            continue;  // 分母为0，跳过该解
        }
        double b = -(x1 * y3 - x3 * y1) / denominator;

        // 构造完整E矩阵（归一化后）
        Eigen::Matrix<double, 3, 3> E_norm;
        E_norm.block<3, 2>(0, 0) = x_norm;  // 前两列
        E_norm(1, 2) = b;                   // e8 = b
        E_norm(2, 2) = c;                   // e9 = c

        // 计算e7 = -(E(1)*E(5)*E(9) - E(1)*E(6)*E(8) - E(2)*E(4)*E(9) + E(3)*E(4)*E(8))/(E(2)*E(6) - E(3)*E(5))
        double numerator = -(E_norm(0, 0) * E_norm(1, 1) * E_norm(2, 2) - E_norm(0, 0) * E_norm(2, 1) * E_norm(1, 2) - E_norm(1, 0) * E_norm(0, 1) * E_norm(2, 2) + E_norm(2, 0) * E_norm(0, 1) * E_norm(1, 2));
        double denom_e7 = E_norm(1, 0) * E_norm(2, 1) - E_norm(2, 0) * E_norm(1, 1);
        if (std::abs(denom_e7) < 1e-12) {
            continue;
        }
        E_norm(0, 2) = numerator / denom_e7;  // e7

        // 反归一化（对应MATLAB的Ef = Ef/fac）
        Eigen::Matrix<double, 3, 3> E = E_norm / fac;
        Ef_list.push_back(E);
    }

    return Ef_list;
}

// 从E32恢复E33(先求解y)
std::vector<Eigen::Matrix<double, 3, 3>> SolveLastColOfEmatrixY(const Eigen::Matrix<double, 3, 2>& x) 
{
    // ========== Step 1: 归一化（对应MATLAB的fac计算） ==========
    // 不需要展平，直接计算
    double mean_abs = x.array().abs().mean();  // 直接计算所有元素绝对值的均值
    double fac = 1.0 / mean_abs;
    Eigen::Matrix<double, 3, 2> x_norm = x * fac;

    // 提取前两列元素（对应e1~e6）
    double e1 = x_norm(0, 0), e2 = x_norm(1, 0), e3 = x_norm(2, 0);
    double e4 = x_norm(0, 1), e5 = x_norm(1, 1), e6 = x_norm(2, 1);

    double k11;
    double k12;
    double k13;
    double k14;

    double k21;
    double k22;
    double k23;
    double k24;

    const double e1_2 = e1 * e1, e1_3 = e1_2 * e1;
    const double e2_2 = e2 * e2, e3_2 = e3 * e3;
    const double e4_2 = e4 * e4, e4_3 = e4_2 * e4;
    const double e5_2 = e5 * e5, e6_2 = e6 * e6;

    const double denom = e1 * e5 - e2 * e4;
    const double denom2 = denom * denom;

    const double common = e1_2 * e5_2 + e1_2 * e6_2 - 2 * e1 * e2 * e4 * e5 - 2 * e1 * e3 * e4 * e6 + e2_2 * e4_2 + e3_2 * e4_2;

    // k11
    k11 = (e1_3 * e5_2 - 2 * e1_2 * e2 * e4 * e5 + e1 * e2_2 * e4_2 - e1 * e2_2 * e6_2 + e1 * e3_2 * e5_2 + 2 * e2_2 * e3 * e4 * e6 - 2 * e2 * e3_2 * e4 * e5) / denom2;

    // k12
    k12 = (2 * e2 * common) / denom2;

    // k13
    k13 = -(e1 * common) / denom2;

    // k14
    k14 = e1_3 + e1 * e2_2 + e1 * e3_2 + e1 * e4_2 - e1 * e5_2 - e1 * e6_2 + 2 * e2 * e4 * e5 + 2 * e3 * e4 * e6;

    // k21
    k21 = (e1_2 * e4 * e5_2 - 2 * e1 * e2 * e4_2 * e5 - 2 * e1 * e2 * e5 * e6_2 + 2 * e1 * e3 * e5_2 * e6 + e2_2 * e4_3 + e2_2 * e4 * e6_2 - e3_2 * e4 * e5_2) / denom2;

    // k22
    k22 = (2 * e5 * common) / denom2;

    // k23
    k23 = -(e4 * common) / denom2;

    // k24
    k24 = e1_2 * e4 + 2 * e1 * e2 * e5 + 2 * e1 * e3 * e6 - e2_2 * e4 - e3_2 * e4 + e4_3 + e4 * e5_2 + e4 * e6_2;

    double g1;
    double g2;
    double g3;

    // 预计算
    const double k11_2 = k11 * k11;
    const double k12_2 = k12 * k12;
    const double k13_2 = k13 * k13;
    const double k21_2 = k21 * k21;
    const double k22_2 = k22 * k22;

    // g1 = k11*(k11*k24 - k14*k21)^2
    const double temp = k11 * k24 - k14 * k21;
    g1 = k11 * temp * temp;

    // g2
    g2 = k11 * (k11 * k14 * k22_2 +
                2 * k13 * k14 * k21_2 +
                k12_2 * k21 * k24 +
                2 * k11_2 * k23 * k24 -
                k11 * k12 * k22 * k24 -
                2 * k11 * k13 * k21 * k24 -
                2 * k11 * k14 * k21 * k23 -
                k12 * k14 * k21 * k22);

    // g3
    g3 = k11 * (k11_2 * k23 * k23 -
                k11 * k12 * k22 * k23 -
                2 * k11 * k13 * k21 * k23 +
                k11 * k13 * k22_2 +
                k12_2 * k21 * k23 -
                k12 * k13 * k21 * k22 +
                k13_2 * k21_2);

    // Eigen::VectorXcd roots = solveQuadraticDirect(g1, g2, g3);
    Eigen::VectorXcd roots = solveQuadraticDirect(g3, g2, g1);
    // std::cout << "多项式根:" << std::endl;
    // for (int i = 0; i < roots.size(); i++) {
    //     std::cout << "根 " << i << ": " << roots(i).real() << " + " << roots(i).imag() << "i" << std::endl;
    // }

    // 筛选正实数根（对应MATLAB的find(ss>0)）
    std::vector<double> valid_roots;
    for (int i = 0; i < roots.size(); ++i) {
        // 检查是否为实数（虚部趋近于0）且大于0
        if (std::abs(roots(i).imag()) < 1e-8 && roots(i).real() > 1e-8) {
            valid_roots.push_back(roots(i).real());
        }
    }

    if (valid_roots.empty()) {
        return {};  // 无有效解
    }

    // 构造cSol = [sqrt(ss); -sqrt(ss)]（对应MATLAB的符号相反解）
    std::vector<double> ySol;
    for (double r : valid_roots) {
        double sqrt_r = std::sqrt(r);
        ySol.push_back(sqrt_r);
        ySol.push_back(-sqrt_r);
    }

    // ========== Step 4: 还原E矩阵第三列（对应MATLAB的for循环） ==========
    std::vector<Eigen::Matrix<double, 3, 3>> Ef_list;
    for (double y : ySol) {
        double x = 0;
        double z = 0;

        x = (k21 * (k13 * y * y + k14) - k11 * (k23 * y * y + k24)) / (k11 * k22 * y - k12 * k21 * y);
        z = (e2 * e6 * x - e3 * e5 * x + e3 * e4 * y - e1 * e6 * y) / (e2 * e4 - e1 * e5);

        // 构造完整E矩阵（归一化后）
        Eigen::Matrix<double, 3, 3> E_norm;
        E_norm.block<3, 2>(0, 0) = x_norm;  // 前两列
        E_norm(0, 2) = x;                   // e9 = c
        E_norm(1, 2) = y;                   // e8 = b
        E_norm(2, 2) = z;                   // e9 = c

        // 反归一化
        Eigen::Matrix<double, 3, 3> E = E_norm / fac;
        Ef_list.push_back(E);
    }

    return Ef_list;
}

// 光心入射光线计算
int MultimediumCalibrator::calculateV0Rays() 
{
    int m = image_points_.size();
    int n = image_points_[0].size();

    std::cout<<"image_points_.size()"<<m<<std::endl;
    std::cout<<"image_points_[0]"<<n<<std::endl;

    for (int i = 0; i < m; i++) {
        std::vector<Eigen::Vector3d> rays;
        for (int j = 0; j < n; j++) {
            std::cout<<image_points_[i][j]<<std::endl;
            Eigen::Vector3d v = pixelToCamRay(image_points_[i][j], fx_, fy_, cx_, cy_);
            rays.push_back(v);
        }
        v0_rays_.push_back(rays);
    }

    for (int i = 0; i < m; i++) {  // m = 图像数量
        // 创建当前图像的三维数据结构
        std::vector<std::vector<Eigen::Vector3d>> img_rays_3d;

        std::cout<<"board_size_.height"<<board_size_.height<<std::endl;

        img_rays_3d.resize(board_size_.height);  // 棋盘格高度
        
        for (int row = 0; row < board_size_.height; row++) {
            img_rays_3d[row].resize(board_size_.width);  // 棋盘格宽度
            
            for (int col = 0; col < board_size_.width; col++) {
                // 计算原始一维索引（假设按行优先存储）
                int idx = row * board_size_.width + col;

                std::cout<<"image_points_[i][idx]"<<image_points_[i][idx]<<std::endl;
                // 从image_points_获取角点并转换为射线
                Eigen::Vector3d v = pixelToCamRay(
                    image_points_[i][idx], 
                    fx_, fy_, cx_, cy_
                );
                img_rays_3d[row][col] = v;
            }
        }
        v0_rays_3d_.push_back(img_rays_3d);
    }


    return 0;
}

// 类构造函数
MultimediumCalibrator::MultimediumCalibrator(cv::Size board_size, float square_size)
    : board_size_(board_size),
      square_size_(square_size),
      total_corners_(board_size.width * board_size.height) 
{
    // 初始化其他参数（相机标定参数焦距、光心）
    fx_ = 0;
    fy_ = 0;
    cx_ = 0;
    cy_ = 0;
}

// 轴估计-11点法
int MultimediumCalibrator::estimateAxisElevenPoints() {
    return 0;
}

Eigen::Vector3d MultimediumCalibrator::getAxisFromE(const Eigen::Matrix<double, 3, 3>& E) {
    Eigen::Vector3d result;
    // 1. 对矩阵A做全SVD分解（U为3x3，包含所有左奇异向量）
    Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3>> svd(
        E,
        Eigen::ComputeFullU | Eigen::ComputeFullV  // 计算完整的U和V
    );

    // 2. 提取奇异值和左奇异向量矩阵U
    // const Eigen::Vector3d& singular_values = svd.singularValues();
    const Eigen::Matrix<double, 3, 3>& U = svd.matrixU();

    if (U.col(2)(2) < 0) {
        result = -U.col(2);
    } else {
        result = U.col(2);
    }

    return result;
}


// 从本质矩阵计算旋转矩阵和平移方向
std::pair<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>>
MultimediumCalibrator::getRotationMatricesFromE(const Eigen::Matrix3d& E,
                                                const Eigen::Vector3d& s) 
{
    // 初始化输出
    std::vector<Eigen::Matrix3d> Rsol(4);
    std::vector<Eigen::Vector3d> sSol(4);
    
    // 对本质矩阵进行 SVD 分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    // Hartley-Zisserman W 矩阵（绕 z 轴旋转 90°）
    Eigen::Matrix3d W;
    W << 0, -1,  0,
         1,  0,  0,
         0,  0,  1;
    
    // 计算两个候选旋转矩阵
    Eigen::Matrix3d R1 = U * W * V.transpose();
    Eigen::Matrix3d R2 = U * W.transpose() * V.transpose();
    
    if (R1.determinant() < 0) {
        R1 = -R1;
    }
    if (R2.determinant() < 0) {
        R2 = -R2;
    }
    
    Eigen::Matrix3d Rsol1;
    Rsol1 = R1;

    Eigen::Matrix3d Rsol2;
    Rsol2 = R2;

    // std::cout<<"E"<<E<<std::endl;
    // std::cout<<"R1"<<R1<<std::endl;
    // std::cout<<"R2"<<R2<<std::endl;
    // std::cout<<"s"<<s<<std::endl;
    
    Rsol[0] = Rsol1;
    sSol[0] = s;
    
    Rsol[1] = Rsol2;
    sSol[1] = s;
    
    Rsol[2] = Rsol1;
    sSol[2] = -s;
    
    Rsol[3] = Rsol2;
    sSol[3] = -s;

    // Rsol[4] = -Rsol1;
    // sSol[4] = -s;
    
    // Rsol[5] = -Rsol2;
    // sSol[5] = -s;
    
    // Rsol[6] = -Rsol1;
    // sSol[6] = s;
    
    // Rsol[7] = -Rsol2;
    // sSol[7] = s;

    return std::make_pair(Rsol, sSol);
}

/**
 * @brief 2D向量叉乘（对应MATLAB的Cross2函数，a×b = a1*b2 - a2*b1）
 * @param a 2D向量
 * @param b 2D向量
 * @return 叉乘标量结果
 */
double Cross2(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a(0) * b(1) - a(1) * b(0);
}

// 计算厚度d0、d1和位移向量 Case2
void MultimediumCalibrator::estimateTzTwoLayerCase2KnownMu(int image_index,
                                                           const Eigen::Matrix3d R,
                                                           const Eigen::Vector3d n,
                                                           const Eigen::Vector3d s,
                                                           const std::vector<int> ids,
                                                           double& d0,
                                                           double& d1,
                                                           Eigen::Vector3d& t) {
}

// 计算厚度d0、d1和位移向量 Case3
void MultimediumCalibrator::estimateTzTwoLayerCase3KnownMu(int image_index,
                                                           const Eigen::Matrix3d R,
                                                           const Eigen::Vector3d n,
                                                           const Eigen::Vector3d s,
                                                           const std::vector<int> ids,
                                                           double& d0,
                                                           double& d1,
                                                           Eigen::Vector3d& t) {
}

// 计算厚度d0和位移向量 Case3(d1作为已知量)(经过校验)
void MultimediumCalibrator::estimateTzTwoLayerCase3KnownMuD1(int image_index,
                                                             const Eigen::Matrix3d R,
                                                             const Eigen::Vector3d n,
                                                             const Eigen::Vector3d s,
                                                             const std::vector<int> ids,
                                                             double& d0,
                                                             Eigen::Vector3d& t,
                                                             double& res) 
{
    // std::cout<<"R"<<R<<std::endl;
    // std::cout<<"s"<<s<<std::endl;
                                                                
    // 玻璃厚度
    double d1 = 10.0;
    Eigen::Matrix3d Rc = alignAxisToZ(n);

    Eigen::Vector3d ss = Rc * s;

    // std::cout<<"ss"<<ss<<std::endl;

    t << ss(1), -ss(0), 0.0;

    // Eigen::Vector3d t_true (0,0,20);
    // std::cout<<"Rc*t_true"<<Rc*t_true<<std::endl;

    // 根据Rc矩阵进行坐标旋转
    int point_nums = 30;
    std::vector<Eigen::Vector3d> XYZ_transformed(point_nums);
    std::vector<Eigen::Vector3d> v0_transformed(point_nums);
    for (int k = 0; k < point_nums; ++k) {
        int j = k;
        Eigen::Vector3d XYZ = {object_points_[image_index][j].x,
                               object_points_[image_index][j].y,
                               object_points_[image_index][j].z};
        XYZ_transformed[k] = Rc * R * XYZ + t;
        // Eigen::Vector3d t_true (0,0,20);
        // XYZ_transformed[k] = Rc * (R * XYZ + t_true);

        Eigen::Vector3d v0 = {v0_rays_[image_index][j](0),
                              v0_rays_[image_index][j](1),
                              v0_rays_[image_index][j](2)};
        v0_transformed[k] = Rc * v0;
    }

    // 构建线性系统
    Eigen::MatrixXd A(point_nums, 2);
    Eigen::VectorXd b(point_nums);
    A.setZero();
    b.setZero();
    Eigen::Vector3d z1(0, 0, 1);   // 归一化坐标系z轴
    Eigen::Vector2d zp(0, 1);      // 辅助基向量
    Eigen::Vector2d n2D(0, 1);     // 二维法向量
    Eigen::Vector2d normal(0, 1);  // 二维法向量

    for (int k = 0; k < point_nums; k++) {
        Eigen::Vector3d XYZ = XYZ_transformed[k];
        Eigen::Vector3d v0 = v0_transformed[k];

        // 计算z2（垂直于z1和v的单位向量）
        Eigen::Vector3d z2 = z1.cross(z1.cross(v0));
        if (z2.norm() > 1e-8) {
            z2.normalize();
            if (z2(0) < 0)
                z2 = -z2;  // 统一x轴正方向
        } else {
            z2 << 1, 0, 0;  // 退化情况，默认x轴
        }

        Eigen::Vector2d vp0;
        vp0 << z2.dot(v0), z1.dot(v0);
        vp0.normalize();

        Eigen::Vector2d vp1;
        Eigen::Vector2d vp2;
        double mu1 = 1;
        double mu2 = 1.5;
        double mu3 = 1.333;
        int tir01;
        int tir12;
        RefractedRay2D(vp0, mu1, mu2, vp1, tir01);
        RefractedRay2D(vp1, mu2, mu3, vp2, tir12);

        // 构建b矩阵
        Eigen::Vector2d u(z2.dot(XYZ), z1.dot(XYZ));
        b(k) = Cross2(vp2, u);

        double c0 = vp0.dot(n2D);
        double c1 = vp1.dot(n2D);

        // 避免除0
        if (fabs(c0) < 1e-8)
            c0 = 1e-8;
        if (fabs(c1) < 1e-8)
            c1 = 1e-8;

        // 构建A矩阵
        A(k, 0) = Cross2(vp2, vp0 / c0);
        A(k, 1) = -Cross2(vp2, zp);
        b(k) -= (Cross2(vp2, vp1 / c1) * d1);
    }

    // 最小二乘求解Ax = b
    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
    d0 = x(0);
    t(2) = x(1);
    t = Rc.inverse()*t;

    Eigen::VectorXd residual = A * x - b;
    res = residual.norm();
    // std::cout << "d0" << d0 << std::endl;
    // std::cout << "res" << res << std::endl;
    // std::cout << "-x(1)" << -x(1) << std::endl;
    // std::cout << "t" << t << std::endl;

}

// 轴估计-8点法
int MultimediumCalibrator::estimateAxisEightPoints() {
    return 0;
}

// 轴估计（单张图像）
int MultimediumCalibrator::estimateAxisEightPointsImage(int image_index,
                                double &d0_closed,
                                Eigen::Vector3d & axis_closed,
                                Eigen::Matrix3d &R_closed,
                                Eigen::Vector3d &t_closed,
                                double &res_closed) 
{
    const int point_nums = 30;
    size_t i = 0;

    // 0. 构建所有点的B矩阵
    B_.clear();
    for (int j = 0; j < total_corners_; j++) {
        Eigen::Matrix<double, 1, 12> B;

        // P：标定板坐标系的角点坐标
        Eigen::Matrix<double, 3, 1> P = {object_points_[image_index][j].x,
                                         object_points_[image_index][j].y,
                                         object_points_[image_index][j].z};

        // v0：穿过图像角点的光线
        Eigen::Matrix<double, 3, 1> v0 = v0_rays_[image_index][j];

        Eigen::Matrix<double, 1, 9> B1 = Eigen::kroneckerProduct(P.transpose(), v0.transpose());
        Eigen::Matrix<double, 1, 3> B2 = v0.transpose();
        B.block(0, 0, 1, 9) = B1;
        B.block(0, 9, 1, 3) = B2;
        B_.push_back(B);
    }

    // 1. 构建单个B矩阵（从B_中随机抽取8个点构建B矩阵）
    std::vector<int> ids;
    std::vector<Eigen::Matrix<double, 1, 12>> B_Ransac = RandomSampleRANSAC_Shuffle(B_, point_nums, ids);
    Eigen::Matrix<double, 30, 9> B_small;
    for (i = 0; i < point_nums; i++) {
        B_small.block(i, 0, 1, 6) = B_Ransac[i].block(0, 0, 1, 6);
        B_small.block(i, 6, 1, 3) = B_Ransac[i].block(0, 9, 1, 3);
    }

    // 2. 对B进行奇异值分解（SVD）
    Eigen::JacobiSVD<Eigen::Matrix<double, 30, 9>> svd(B_small, Eigen::ComputeFullV);
    Eigen::Matrix<double, 9, 9> V = svd.matrixV();

    // 4. 零空间的解向量：对应最小奇异值的右奇异向量（V的最后一列）
    // 因为A是11×12，秩最多11，所以最后一列必在零空间中
    Eigen::Matrix<double, 9, 1> sol = V.col(8);
    Eigen::Vector3d s = sol.block(6, 0, 3, 1);
    Eigen::Matrix<double, 3, 2> E32;
    E32.block(0, 0, 3, 1) = sol.block(0, 0, 3, 1);
    E32.block(0, 1, 3, 1) = sol.block(3, 0, 3, 1);

    // 从E32恢复E33
    std::vector<Eigen::Matrix<double, 3, 3>> E33s = SolveLastColOfEmatrixY(E32);
    // std::vector<Eigen::Matrix<double, 3, 3>> E33s = SolveLastColOfEmatrix(E32);
    if (E33s.empty()) {
        printf("E33恢复为空\n");
        return -1;
    }

    // 进行放缩
    double fac = 1.0;
    const double sqrt2 = sqrt(2.0);
    const double eps = 1e-8;
    for (size_t ii = 0; ii < E33s.size(); ++ii) {  // 遍历每一页
        // 提取当前页并展平为向量
        Eigen::Matrix3d tt_mat = E33s[ii];
        Eigen::VectorXd tt_vec = Eigen::Map<Eigen::VectorXd>(tt_mat.data(), tt_mat.size());

        // 计算范数和缩放系数
        double tt_norm = tt_vec.norm();
        if (tt_norm < eps) {
            fac = 1.0;
        } else {
            fac = sqrt2 / tt_norm;
        }

        // 归一化当前页
        E33s[ii] *= fac;
    }
    s *= fac;

    // 从E33恢复A和R
    std::vector<Eigen::Vector3d> axises;
    std::vector<Eigen::Matrix3d> rotations;
    std::vector<Eigen::Vector3d> ss;

    for (i = 0; i < E33s.size(); i++) {
        Eigen::Matrix3d E33 = E33s[i];

        // 恢复A轴
        Eigen::Vector3d axis = getAxisFromE(E33);
        axises.push_back(axis);

        // 恢复R矩阵
        auto [Rsol, sSol] = getRotationMatricesFromE(E33, s);
        for (size_t k = 0; k < Rsol.size(); k++) {
            rotations.push_back(Rsol[k]);
            ss.push_back(sSol[k]);
        }
    }

    // 估计每组解的厚度和距离参数
    std::vector<EstimationResult> d0_ress;
    d0_ress.clear();
    for (size_t i = 0; i < rotations.size(); i++) {
        double d0;
        Eigen::Vector3d t;
        double res;
        estimateTzTwoLayerCase3KnownMuD1(image_index, rotations[i], axises[i / 4], ss[i], ids, d0, t, res);
        
        if(t(2) < 0){
            continue;
        }

        d0_ress.emplace_back(d0,res,axises[i / 4],rotations[i],t);
    }

    // 使用 std::min_element 找到 residual 最小的元素
    auto min_it = std::min_element(d0_ress.begin(), d0_ress.end(),
        [](const EstimationResult& a, const EstimationResult& b) {
            return a.residual < b.residual;  // 使用 residual 成员
        });

    if (min_it != d0_ress.end()) {
        // double best_d0 = min_it->d0;           // 使用 d0 成员
        // double min_res = min_it->residual;     // 使用 residual 成员
        
        // std::cout << "Best d0: " << best_d0 
        //         << " with residual: " << min_res << std::endl;
        
        // // // 可选：打印更多信息
        // std::cout << "Best axis: " << min_it->axis.transpose() << std::endl;
        // std::cout << "Best translation: " << min_it->translation.transpose() << std::endl;

        d0_closed = min_it->d0;
        axis_closed =  min_it->axis;
        R_closed =  min_it->rotation;
        t_closed =  min_it->translation;
        res_closed =  min_it->residual;
    }

    return 0;
}

// 轴估计（所有图像）
int MultimediumCalibrator::estimateAxisEightPointsImages(void) 
{

    d0_closed_.clear();
    axis_closed_.clear();
    R_closed_.clear();
    t_closed_.clear();

    d0_closed_.resize(image_nums_);
    axis_closed_.resize(image_nums_);
    R_closed_.resize(image_nums_);
    t_closed_.resize(image_nums_);
 
    for (int i = 0; i < image_nums_; i++) {
        double d0_closed = 0;
        Eigen::Vector3d axis_closed = Eigen::Vector3d::Zero();
        Eigen::Matrix3d R_closed = Eigen::Matrix3d::Zero();
        Eigen::Vector3d t_closed = Eigen::Vector3d::Zero();
        double res = 0;

        estimateAxisEightPointsImage(i,
                                    d0_closed,
                                    axis_closed,
                                    R_closed,
                                    t_closed,
                                    res);

        d0_closed_[i] = d0_closed;
        axis_closed_[i] = axis_closed;
        R_closed_[i] = R_closed;
        t_closed_[i] = t_closed;

        std::cout<<" i: "<<i<<std::endl
                <<" d0_closed: "<<d0_closed<<std::endl
                <<" axis_closed: "<<axis_closed<<std::endl
                <<" res: "<<res<<std::endl
                <<" R_closed: "<<R_closed<<std::endl
                <<" t_closed: "<<t_closed<<std::endl;
    }

    return 0;
}


// 距离估计
int MultimediumCalibrator::estimateThickness() {
    return 0;
}

//
int MultimediumCalibrator::getObjectPoints(const std::vector<std::vector<cv::Point3f>>& object_points) {
    object_points_ = object_points;
    return 0;
}

//
int MultimediumCalibrator::getImagePoints(const std::vector<std::vector<cv::Point2f>>& image_points) {
    image_points_ = image_points;
    return 0;
}

// 生成世界坐标系3D点（棋盘格平面为z=0）
void MultimediumCalibrator::generateObjectPoints(std::vector<cv::Point3f>& obj_points) {
    obj_points.clear();
    // 按行优先顺序生成3D点（x,y,z），z=0
    for (int y = 0; y < board_size_.height; ++y) {
        for (int x = 0; x < board_size_.width; ++x) {
            obj_points.emplace_back(
                x * square_size_,  // x坐标（列方向）
                y * square_size_,  // y坐标（行方向）
                0.0f               // z坐标（平面）
            );
        }
    }
}
 
// 添加图像并提取角点
bool MultimediumCalibrator::addImage(const cv::Mat& img) {
    if (img.empty()) {
        std::cerr << "错误：输入图像为空！" << std::endl;
        return false;
    }

    img_size_ = img.size();

    // 转换为灰度图（calib3d角点检测需要灰度图）
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 1) {
        gray = img.clone();
    } else {
        std::cerr << "错误：图像通道数不支持（仅支持1或3通道）！" << std::endl;
        return false;
    }

    // 1. 粗检测角点（calib3d的findChessboardCorners）
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(
        gray,
        board_size_,
        corners,
        cv::CALIB_CB_ADAPTIVE_THRESH |  // 自适应阈值
            cv::CALIB_CB_NORMALIZE_IMAGE);

    if (!found || corners.size() != static_cast<size_t>(total_corners_)) {
        std::cerr << "警告：未检测到完整的棋盘格角点（跳过该图像）！" << std::endl;
        return false;
    }

    // 2. 亚像素精确化（calib3d的cornerSubPix）
    cv::cornerSubPix(
        gray,
        corners,
        cv::Size(11, 11),  // 搜索窗口大小
        cv::Size(-1, -1),  // 死区（无死区）
        cv::TermCriteria(  // 终止条件：30次迭代或0.001像素精度
            cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
            30, 0.001));

    // 3. 生成对应世界坐标并保存
    std::vector<cv::Point3f> obj_points;
    generateObjectPoints(obj_points);
    object_points_.push_back(obj_points);  // 世界坐标（固定）
    image_points_.push_back(corners);      // 图像坐标（检测结果）

    // // 输出角点坐标
    // for (int i = 0; i < obj_points.size(); i++) {
    //     std::cout << i << std::endl;
    //     std::cout << corners[i].x << "," << corners[i].y << std::endl;
    //     std::cout << obj_points[i].x << "," << obj_points[i].y << "," << obj_points[i].z << std::endl;
    // }

    // 可视化角点（可选）
    cv::Mat img_with_corners = img.clone();
    cv::drawChessboardCorners(img_with_corners, board_size_, corners, found);
    cv::namedWindow("角点检测结果", cv::WINDOW_NORMAL);
    cv::imshow("角点检测结果", img_with_corners);
    cv::imwrite("./conner.png", img_with_corners);
    cv::waitKey(0);

    // std::cout << "成功添加图像，累计 " << image_points_.size() << " 张有效图像" << std::endl;
    return true;
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
int MultimediumCalibrator::loadImagePaths(const std::string& foldername) {
    image_paths_.clear();
    DIR* dir = opendir(foldername.c_str());
    if (!dir)
        return -1;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) {
            image_paths_.push_back(foldername + "/" + entry->d_name);
        }
    }
    closedir(dir);

    // ========== 关键：按文件名中的数字升序排序 ==========
    std::sort(image_paths_.begin(), image_paths_.end(),
              [](const std::string& a, const std::string& b) {
                  int num_a = extractNumberFromFilename(a);
                  int num_b = extractNumberFromFilename(b);
                  return num_a < num_b;  // 升序（1,2,3...25），降序则改为 num_a > num_b
              });
    image_nums_ = image_paths_.size();

    return image_paths_.size();
}

// 加载标定图像
int MultimediumCalibrator::loadImages() {
    for (const auto& path : image_paths_) {
        cv::Mat img = cv::imread(path);
        std::cout << path << std::endl;
        if (img.empty()) {
            std::cerr << "无法读取图像：" << path << std::endl;
            continue;
        }

        // 图像畸变校正
        cv::Mat K = (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_,  // fx=焦距x, cx=主点x
                     0.0, fy_, cy_,                            // fy=焦距y, cy=主点y
                     0.0, 0.0, 1.0);

        cv::Mat D = (cv::Mat_<double>(1, 5) << k1_, k2_, p1_, p2_, k3_);
        cv::Mat undistorted_img1;
        undistort(img, undistorted_img1, K, D);

        addImage(undistorted_img1);  // 提取角点并添加
    }

    return 0;
}

void MultimediumCalibrator::getReprojectionErrorCase3(const Eigen::Vector3d axis,
                                                        const Eigen::Matrix3d R,
                                                        const Eigen::Vector3d t,
                                                        const double d0,
                                                        const double d1,
                                                        const int id,
                                                        double &error)
{
    error = 0;
    // 先根据封闭解计算的R和t计算相机坐标系下的角点坐标
    std::vector<Eigen::Vector3d> camera_points;
    for (size_t i = 0; i < object_points_[id].size(); i++) {
        Eigen::Vector3d P = Eigen::Vector3d::Zero();
        P(0) = object_points_[id][i].x;
        P(1) = object_points_[id][i].y;
        P(2) = object_points_[id][i].z;

        Eigen::Vector3d camera_point = R * P + t;
        camera_points.push_back(camera_point);
    }

    // // 利用正向投影计算归一化平面坐标，
    // std::vector<Eigen::Vector3d> reproject_rays;
    // for (size_t i = 0; i < camera_points.size(); i++) {
    //     Eigen::Vector3d camera_point = camera_points[i];
    //     Eigen::Vector3d reproject_ray = SolveForwardProjectionCase3(d, d2, n, mu, rotated);
    //     ray.normalize();
    //     reproject_rays.push_back(ray);
    // }
    
    // // 最后计算归一化平面坐标误差（空气-玻璃-水介质）
    // for (size_t i = 0; i < reproject_rays.size(); i++) {
    //     Eigen::Vector3d reproject_ray = reproject_rays[i];
    //     Eigen::Vector3d image_ray = v0_rays_[id][i];

    //     Eigen::Vector3d diff = reproject_ray-image_ray;
    //     diff.squaredNorm();
    // }
}

// tools.cpp
#include "tools.h"

// ========== 计算将轴对准到（0，0，1）的旋转矩阵 ==========
Eigen::Matrix3d alignAxisToZ(const Eigen::Vector3d& axis) 
{
    Eigen::Vector3d n = axis.normalized();
    Eigen::Vector3d target(0, 0, 1);

    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(n, target);
    return q.toRotationMatrix();
}

// ========== 折射光线计算函数（2D） ==========
void RefractedRay2D(const Eigen::Vector2d& incident,
                    double mu1,
                    double mu2,
                    Eigen::Vector2d& refracted,
                    int& tir) 
{
    // 1. 初始化
    Eigen::Vector2d normal(0, -1);
    tir = 0;

    // 2. 归一化入射光线
    double incident_norm = incident.norm();
    if (incident_norm < 1e-10) {
        std::cout<<incident<<std::endl;
        throw std::invalid_argument("RefractedRay: Incident ray has zero length!");
    }
    Eigen::Vector2d vi = incident / incident_norm;

    // 3. 计算入射角余弦
    double cos_theta_i = -vi.dot(normal);

    // 4. 处理法向量方向
    Eigen::Vector2d effective_normal = normal;
    if (cos_theta_i < 0) {
        effective_normal = -normal;
        cos_theta_i = -cos_theta_i;
    }

    // 5. 限制在有效范围
    cos_theta_i = std::max(0.0, std::min(1.0, cos_theta_i));

    // 6. 计算折射率比
    double eta = mu1 / mu2;

    // 7. 计算折射角的正弦平方
    double sin2_theta_t = eta * eta * (1.0 - cos_theta_i * cos_theta_i);

    // 8. 检查全反射
    if (sin2_theta_t > 1.0) {
        tir = 1;
        refracted = Eigen::Vector2d::Zero();
        return;
    }

    // 9. 计算折射角余弦
    double cos_theta_t = std::sqrt(1.0 - sin2_theta_t);

    // 10. 斯涅尔定律的向量形式
    refracted = eta * vi + (eta * cos_theta_i - cos_theta_t) * effective_normal;

    // 11. 归一化
    refracted.normalize();
}

// ========== 折射光线计算函数（3D） ==========
void RefractedRay3D(const Eigen::Vector3d& incident,
                    const Eigen::Vector3d& normal,
                    double mu1,
                    double mu2,
                    Eigen::Vector3d& refracted,
                    int& tir) 
{
    // 归一化输入向量
    Eigen::Vector3d I = incident.normalized();  // 入射方向
    Eigen::Vector3d N = normal.normalized();    // 法向量

    // 计算 cos(θi) = -I·N
    // 注意：这里用负号，因为入射光线应该朝向界面（与法向量反向）
    double cos_theta_i = -I.dot(N);

    // 如果 cos_theta_i < 0，说明光线从另一侧入射，翻转法向量
    if (cos_theta_i < 0) {
        N = -N;
        cos_theta_i = -cos_theta_i;
    }

    // 折射率比 η = n1/n2
    double eta = mu1 / mu2;

    // 判别式：k = 1 - η²·(1 - cos²(θi))
    double k = 1.0 - eta * eta * (1.0 - cos_theta_i * cos_theta_i);

    if (k < 0.0) {
        // 全反射
        tir = 1;
        refracted = Eigen::Vector3d::Zero();
    } else {
        // 正常折射
        tir = 0;

        // 斯涅尔定律的向量形式：
        // T = η·I + (η·cos(θi) - √k)·N
        double cos_theta_t = std::sqrt(k);

        refracted = eta * I + (eta * cos_theta_i - cos_theta_t) * N;

        // 归一化折射光线
        refracted.normalize();
    }
}

// 已知像素点（支持亚像素）、相机内参计算归一化光线向量
Eigen::Vector3d pixelToCamRay(const cv::Point2f& pixel,
                              double fx,
                              double fy,
                              double cx,
                              double cy) 
{
    // 1. 提取像素坐标（u = pixel.x, v = pixel.y）
    float u = pixel.x;
    float v = pixel.y;

    // 2. 计算归一化图像坐标（x', y'）
    double x_prime = (u - cx) / fx;
    double y_prime = (v - cy) / fy;

    // 3. 光心坐标系下的光线向量（归一化）
    Eigen::Vector3d dir_cam(x_prime, y_prime, 1.0);
    dir_cam.normalize();  // 转为单位向量，方便方向计算

    return dir_cam;
}

// 求解12次多项式根
std::vector<std::complex<double>> solvePolynomialTwelve(const Eigen::VectorXd& coeffs) 
{
    int n = coeffs.size() - 1;

    // 找到第一个非零系数
    int start = 0;
    while (start < n && abs(coeffs(start)) < 1e-15) {
        start++;
    }

    if (start >= n) {
        return std::vector<std::complex<double>>();
    }

    int degree = n - start;
    if (degree == 0) {
        return std::vector<std::complex<double>>();
    }

    // 构建伴随矩阵
    Eigen::MatrixXd companion = Eigen::MatrixXd::Zero(degree, degree);

    double leadCoeff = coeffs(start);

    // 第一行：归一化的系数
    for (int i = 0; i < degree; i++) {
        companion(0, i) = -coeffs(start + i + 1) / leadCoeff;
    }

    // 次对角线
    for (int i = 1; i < degree; i++) {
        companion(i, i - 1) = 1.0;
    }

    // 计算特征值
    Eigen::EigenSolver<Eigen::MatrixXd> solver(companion);
    auto eigenvalues = solver.eigenvalues();

    std::vector<std::complex<double>> roots;
    for (int i = 0; i < eigenvalues.size(); i++) {
        roots.push_back(eigenvalues(i));
    }

    return roots;
}


// ========== 求解前向投影Case3（水+玻璃+空气），已知相机坐标系下的3D点，计算归一化像素坐标（三维） ==========
Eigen::Vector3d ForwardProjectionCase3(double d, 
                                        double d2, 
                                        const Eigen::Vector3d& n, 
                                        const Eigen::Vector2d& mu, 
                                        const Eigen::Vector3d& p) 
{
    double mu1 = mu(0);
    double mu2 = mu(1);

    Eigen::Vector3d M(0, 0, 1);

    // 计算折射平面 (POR)
    Eigen::Vector3d POR = n.cross(p);
    double por_norm = POR.norm();
    if (por_norm < 1e-10) {
        std::cout << "Point is parallel to normal" << std::endl;
        return M;
    }
    POR = POR / por_norm;

    // 定义坐标系 z1, z2
    Eigen::Vector3d z1 = -n;
    z1 = z1 / z1.norm();

    Eigen::Vector3d z2 = POR.cross(z1);
    z2 = z2 / z2.norm();

    // 将3D点投影到POR上
    double v = p.dot(z1);
    double u = p.dot(z2);

    // 计算多项式系数
    double mu1_2 = mu1 * mu1;
    double mu2_2 = mu2 * mu2;
    double d_2 = d * d;
    double d_4 = d_2 * d_2;
    double d_8 = d_4 * d_4;
    double u_2 = u * u;
    double u_3 = u_2 * u;
    double u_4 = u_2 * u_2;

    double term1 = mu1_2 - 1.0;
    double term2 = mu2_2 - 1.0;
    double term1_2 = term1 * term1;
    double term2_2 = term2 * term2;

    double dd = d - d2;
    double dd_2 = dd * dd;
    double dv = d2 - v;
    double dv_2 = dv * dv;

    // 计算中间项 A, B, C
    double A = term2 * (u_2 * term1 + d_2 * mu1_2) -
               term2 * dd_2 -
               term1 * dv_2 +
               d_2 * mu2_2 * term1;

    double B = d_2 * mu2_2 * (u_2 * term1 + d_2 * mu1_2) -
               d_2 * mu2_2 * dd_2 -
               d_2 * mu1_2 * dv_2 +
               d_2 * mu1_2 * u_2 * term2;

    double C = 2.0 * d_2 * mu1_2 * u * term2 +
               2.0 * d_2 * mu2_2 * u * term1;

    Eigen::VectorXd coeffs(13);

    // s1
    coeffs(0) = term1_2 * term2_2;

    // s2
    coeffs(1) = -4.0 * u * term1_2 * term2_2;

    // s3
    coeffs(2) = 4.0 * u_2 * term1_2 * term2_2 +
                2.0 * term1 * term2 * A;

    // s4
    coeffs(3) = -2.0 * term1 * term2 * C -
                4.0 * u * term1 * term2 * A;

    // s5
    coeffs(4) = A * A +
                2.0 * term1 * term2 * B -
                4.0 * term1 * term2 * dd_2 * dv_2 +
                4.0 * u * term1 * term2 * C;

    // s6
    coeffs(5) = -2.0 * C * A -
                4.0 * u * term1 * term2 * B -
                4.0 * d_4 * mu1_2 * mu2_2 * u * term1 * term2;

    // s7
    coeffs(6) = 2.0 * A * B +
                C * C -
                4.0 * dd_2 * (d_2 * mu1_2 * term2 + d_2 * mu2_2 * term1) * dv_2 +
                10.0 * d_4 * mu1_2 * mu2_2 * u_2 * term1 * term2;

    // s8
    coeffs(7) = -2.0 * C * B -
                4.0 * d_4 * mu1_2 * mu2_2 * u * A -
                4.0 * d_4 * mu1_2 * mu2_2 * u_3 * term1 * term2;

    // s9
    coeffs(8) = B * B +
                2.0 * d_4 * mu1_2 * mu2_2 * u_2 * A -
                4.0 * d_4 * mu1_2 * mu2_2 * dd_2 * dv_2 +
                4.0 * d_4 * mu1_2 * mu2_2 * u * C;

    // s10
    coeffs(9) = -2.0 * d_4 * mu1_2 * mu2_2 * u_2 * C -
                4.0 * d_4 * mu1_2 * mu2_2 * u * B;

    // s11
    coeffs(10) = 4.0 * d_8 * mu1_2 * mu1_2 * mu2_2 * mu2_2 * u_2 +
                 2.0 * d_4 * mu1_2 * mu2_2 * u_2 * B;

    // s12
    coeffs(11) = -4.0 * d_8 * mu1_2 * mu1_2 * mu2_2 * mu2_2 * u_3;

    // s13
    coeffs(12) = d_8 * mu1_2 * mu1_2 * mu2_2 * mu2_2 * u_4;

    // 求解多项式
    std::vector<std::complex<double>> sol = solvePolynomialTwelve(coeffs);

    if (sol.empty()) {
        std::cout << "No solution found" << std::endl;
        return M;
    }

    // 找到实数解
    std::vector<double> realSols;
    for (const auto& s : sol) {
        if (abs(s.imag()) < 1e-6) {
            realSols.push_back(s.real());
        }
    }

    if (realSols.empty()) {
        std::cout << "No real solution" << std::endl;
        return M;
    }

    // 验证每个解
    // Vector2d Normal(0, 1);
    Eigen::Vector2d Normal(0, -1);

    for (size_t ii = 0; ii < realSols.size(); ii++) {
        double x = realSols[ii];
        Eigen::Vector2d vi(x, d);

        // 第一次折射
        int tir = 0;
        Eigen::Vector2d v2;
        v2.setZero();
        RefractedRay2D(vi, 1.0, mu1, v2, tir);

        if (v2.norm() < 1e-10) {
            continue;
        }

        // 计算第二个界面上的点 q2
        double dot_product = v2.dot(Normal);
        if (abs(dot_product) < 1e-10) {
            continue;
        }

        Eigen::Vector2d q2 = vi + (d - d2) * v2 / dot_product;

        // 第二次折射
        Eigen::Vector2d v3;
        v3.setZero();

        RefractedRay2D(v2, mu1, mu2, v3, tir);

        if (v3.norm() < 1e-10) {
            continue;
        }

        // 验证
        Eigen::Vector2d vrd(u, v);
        vrd = vrd - q2;

        double e = abs(vrd(0) * v3(1) - vrd(1) * v3(0));

        if (e < 1e-4) {
            M = x * z2 + d * z1;
            return M;
        }
    }

    std::cout << "No valid solution after verification" << std::endl;
    return M;
}


// ========== 求解反向投影Case3（水+玻璃+空气），已知像素坐标，计算水界面出射点坐标和最终折射光线 ==========
void backwardProjectionCase3(const double d0,
                            const double d1,
                            const Eigen::Vector3d axis,
                            const Eigen::Vector3d mu,
                            const Eigen::Vector3d v0_ray,
                            Eigen::Vector3d& p1,
                            Eigen::Vector3d& v2_ray)
{
    double mu0 = mu(0);     // 空气折射率
    double mu1 = mu(1);     // 玻璃折射率
    double mu2 = mu(2);     // 水折射率

    // 统一到法向（0，0，1坐标系）
    Eigen::Matrix3d Rc = alignAxisToZ(axis);
    // std::cout<<Rc*axis<<std::endl;
    // std::cout<<Rc<<std::endl;

    Eigen::Vector3d v0_ray_Rc = Rc*v0_ray;

    // 计算折射光线
    Eigen::Vector3d v1_ray_Rc;
    Eigen::Vector3d v2_ray_Rc;
    Eigen::Vector3d normal(0,0,1);

    int tir01;
    int tir12;
    RefractedRay3D(v0_ray_Rc,normal,mu0,mu1,v1_ray_Rc,tir01);
    RefractedRay3D(v1_ray_Rc,normal,mu1,mu2,v2_ray_Rc,tir12);

    // 计算出射点
    Eigen::Vector3d p0_Rc = (d0/v0_ray_Rc(2))*v0_ray_Rc;
    Eigen::Vector3d p1_Rc = p0_Rc + (d1/v1_ray_Rc(2))*v1_ray_Rc;

    // std::cout<<(Rc.transpose()*p0_Rc).transpose()<<std::endl;
    // std::cout<<"p1_Rc"<<p1_Rc.transpose()<<std::endl;

    // 统一到相机坐标系
    v2_ray = Rc.transpose()*v2_ray_Rc;
    p1 = Rc.transpose()*p1_Rc;
}

// ========== 求解反向投影Case3（水+玻璃+空气），已知像素坐标，计算水界面出射点坐标和最终折射光线 ==========
void backwardProjectionCase3P0(const double d0,
                            const double d1,
                            const Eigen::Vector3d axis,
                            const Eigen::Vector3d mu,
                            const Eigen::Vector3d v0_ray,
                            Eigen::Vector3d& p0,
                            Eigen::Vector3d& p1,
                            Eigen::Vector3d& v2_ray)
{
    double mu0 = mu(0);     // 空气折射率
    double mu1 = mu(1);     // 玻璃折射率
    double mu2 = mu(2);     // 水折射率

    // 统一到法向（0，0，1坐标系）
    Eigen::Matrix3d Rc = alignAxisToZ(axis);
    // std::cout<<Rc*axis<<std::endl;
    // std::cout<<Rc<<std::endl;

    Eigen::Vector3d v0_ray_Rc = Rc*v0_ray;

    // 计算折射光线
    Eigen::Vector3d v1_ray_Rc;
    Eigen::Vector3d v2_ray_Rc;
    Eigen::Vector3d normal(0,0,1);

    int tir01;
    int tir12;
    RefractedRay3D(v0_ray_Rc,normal,mu0,mu1,v1_ray_Rc,tir01);
    RefractedRay3D(v1_ray_Rc,normal,mu1,mu2,v2_ray_Rc,tir12);

    // 计算出射点
    Eigen::Vector3d p0_Rc = (d0/v0_ray_Rc(2))*v0_ray_Rc;
    Eigen::Vector3d p1_Rc = p0_Rc + (d1/v1_ray_Rc(2))*v1_ray_Rc;

    // std::cout<<(Rc.transpose()*p0_Rc).transpose()<<std::endl;
    // std::cout<<"p1_Rc"<<p1_Rc.transpose()<<std::endl;

    // 统一到相机坐标系
    v2_ray = Rc.transpose()*v2_ray_Rc;
    p1 = Rc.transpose()*p1_Rc;
    p0 = Rc.transpose()*p0_Rc;
}

// ========== Eigen::Matrix3d → 轴角向量==========
Eigen::Vector3d rotationMatrixToAngleAxis(const Eigen::Matrix3d& R) 
{
    Eigen::AngleAxisd angle_axis_eigen(R);
    Eigen::Vector3d axis_angle_vec = angle_axis_eigen.angle() * angle_axis_eigen.axis();
    return axis_angle_vec;
}


// ========== 空间三维点计算（中点法）==========
void midPointReconstruction(const Eigen::Vector3d& left_center,
                            const Eigen::Vector3d& left_ray,
                            const Eigen::Vector3d& right_center,
                            const Eigen::Vector3d& right_ray,
                            Eigen::Vector3d& mid_point,
                            double* reprojection_error) 
{
    // 归一化方向向量
    Eigen::Vector3d d1 = left_ray.normalized();
    Eigen::Vector3d d2 = right_ray.normalized();

    // 连接向量
    Eigen::Vector3d w0 = left_center - right_center;

    // 计算点积
    double a = d1.dot(d1);  // = 1
    double b = d1.dot(d2);
    double c = d2.dot(d2);  // = 1
    double d = d1.dot(w0);
    double e = d2.dot(w0);

    // 判别式
    double denom = a * c - b * b;

    const double EPSILON = 1e-10;

    if (std::abs(denom) < EPSILON) {
        // 光线平行
        mid_point = left_center + d1;
        if (reprojection_error) {
            *reprojection_error = std::numeric_limits<double>::infinity();
        }
        return;
    }


    // 计算最近点对的参数
    double t1 = (b * e - c * d) / denom;
    double t2 = (a * e - b * d) / denom;

    // 确保参数为正（光线向前）
    if (t1 < 0)
        t1 = 0;
    if (t2 < 0)
        t2 = 0;

    // 计算最近点
    Eigen::Vector3d closest1 = left_center + t1 * d1;
    Eigen::Vector3d closest2 = right_center + t2 * d2;

    // 中点即为重建的三维点
    mid_point = 0.5 * (closest1 + closest2);
    // std::cout<<"mid_point"<<mid_point<<std::endl;
    // std::cout<<"closest2"<<closest2<<std::endl;
    // std::cout<<"closest1"<<closest1<<std::endl;

    // 计算重投影误差
    if (reprojection_error) {
        Eigen::Vector3d diff = closest2 - closest1;
        *reprojection_error = diff.norm();
    }
}

/**
 * @brief 判断两条光线是否满足对极约束
 * 对极约束的数学表达：
 * 两条光线、两个相机中心共面 <=> d2 · (baseline × d1) = 0
 * @param left_point 相机1的出射点
 * @param left_ray 相机1的光线方向（归一化）
 * @param right_point 相机2的出射点
 * @param right_ray 相机2的光线方向（归一化）
 * @return 
 */
double checkEpipolarConstraint(const Eigen::Vector3d& left_point,
                            const Eigen::Vector3d& left_ray,
                            const Eigen::Vector3d& right_point,
                            const Eigen::Vector3d& right_ray)
{
    // ========== 1. 计算基线向量 ==========
    Eigen::Vector3d baseline = right_point - left_point;
    
    // ========== 2. 计算叉积 baseline × d1 ==========
    Eigen::Vector3d crossProduct = baseline.cross(left_ray);
    
    // ========== 3. 计算混合积（标量三重积）==========
    // constraint = d2 · (baseline × d1)
    double constraintValue = right_ray.dot(crossProduct);
    
    // // ========== 4. 判断是否接近零 ==========
    // bool satisfied = std::abs(constraintValue) < threshold;

    return constraintValue;
}

/**
 * @brief 计算离散外极曲线，右相机外极点在左相机显示
 * @param left_point 相机1的出射点
 * @param left_ray 相机1的光线方向（归一化）
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
                        Eigen::Vector3d n_left)
{
    result_points.clear();

    // 1. 折射率设置
    double mu1 = 1.5;
    double mu2 = 1.333;
    Eigen::Vector2d mu2D(mu1, mu2);

    // 准备可视化图像
    cv::Mat imgRgb;
    if (srcImg.channels() == 1) {
        cv::cvtColor(srcImg, imgRgb, cv::COLOR_GRAY2BGR);
    } else {
        imgRgb = srcImg.clone();
    }

    // 2. 法向朝向调整
    water_ray_right.normalize();
    n_left.normalize();
    if(n_left(2) > 0)
    {
        n_left = -n_left;
    }

    // 3. 生成离散空间点
    if (maxdistance <= mindistance || step_size <= 0) return;
    int total_nums = static_cast<int>((maxdistance - mindistance) / step_size);
    std::vector<Eigen::Vector3d> eipolar_3Dpoints;
    for(int i = 0; i < total_nums; i++)
    {
        double dist = mindistance + i * step_size;
        Eigen::Vector3d eipolar_3Dpoint = water_point_right + water_ray_right * dist;
        eipolar_3Dpoints.push_back(eipolar_3Dpoint);
    }

    // 4. 遍历采样点并投影
    for (size_t i = 0; i < eipolar_3Dpoints.size(); i++) 
    {
        Eigen::Vector3d vec = ForwardProjectionCase3(d0_left, d0_left + d1_left, n_left, mu2D, eipolar_3Dpoints[i]);
        vec.normalize();

        Eigen::Vector2d pixel; 
        pixel(0) = (vec.x() / vec.z()) * fx + cx;
        pixel(1) = (vec.y() / vec.z()) * fy + cy;
        result_points.push_back(pixel);

        // std::cout<<pixel(0)<<","<<pixel(1)<<std::endl;
    }

    // 5. 绘制离散点
    for(int k = 0; k < result_points.size(); k++)
    {
        cv::Point center(result_points[k](0), result_points[k](1));
        cv::circle(imgRgb, center, 2, cv::Scalar(0, 0, 255), -1);
    }
    imgline = imgRgb;

    // cv::namedWindow("imageline_right", cv::WINDOW_NORMAL);
    // cv::imshow("imageline_right", imgline);
    // cv::waitKey(0);

    return ;
}

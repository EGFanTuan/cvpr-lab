# 计算机视觉与模式识别实验报告

## 实验目的
- 熟悉图像处理的基本操作，OpenCV的基本使用；
- 理解边缘检测的基本概念，掌握Canny边缘检测算法的原理；
- 理解霍夫变换的原理，掌握霍夫圆变换检测圆形的算法；
- 学习自主实现经典图像处理算法，加深对算法原理的理解。

## 实验内容
- 编写一个钱币定位系统，能够检测出输入图像中各个钱币的边缘，同时给出各个钱币的圆心坐标与半径。
- 可直接调用OpenCV的Canny与HoughCircle算法完成系统设计。
- 推荐自主实现Canny与Hough算法（加分项）。

## 实验过程

### 1. 算法整体流程

钱币定位系统的主要流程如下：

```
输入图像 → 灰度转换 → 高斯模糊 → 边缘检测 → 霍夫圆变换 → 输出结果
```

#### 各模块功能说明

| 模块 | 功能 | 使用的函数/算法 |
|------|------|----------------|
| 灰度转换 | 将彩色图像转换为灰度图 | cvtColor |
| 高斯模糊 | 降噪，减少边缘检测的伪影 | m_GaussianBlur / GaussianBlur |
| 边缘检测 | 提取图像中的边缘信息 | m_edgeDetect / Canny |
| 霍夫圆变换 | 在边缘图像中检测圆形 | m_houghCircle / HoughCircles |

---

### 2. 高斯滤波的实现

#### 2.1 算法原理

高斯滤波是一种线性平滑滤波技术，利用高斯函数的权重对图像进行加权平均，从而实现降噪效果。对于5×5的高斯核，权重分布遵循二维高斯函数：

$$G(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$$

#### 2.2 实现代码

```cpp
auto static m_CreateGaussianKernel(int ksize, double sigma) -> cv::Mat {
    cv::Mat kernel(ksize, ksize, CV_64F);
    int halfSize = ksize / 2;
    double sum = 0.0;

    for (int y = -halfSize; y <= halfSize; y++) {
        for (int x = -halfSize; x <= halfSize; x++) {
            double g = (1 / (2 * CV_PI * sigma * sigma)) *
                       exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel.at<double>(y + halfSize, x + halfSize) = g;
            sum += g;
        }
    }

    for(int y = 0; y < ksize; y++) {
        for(int x = 0; x < ksize; x++) {
            kernel.at<double>(y, x) /= sum;  // 归一化
        }
    }
    return kernel;
}
```

#### 2.3 卷积实现

使用零填充对图像进行卷积操作：

```cpp
cv::Mat padded = cv::Mat::zeros(src.rows() + 4, src.cols() + 4, CV_64F);
for(int y = 0; y < src.rows(); y++) {
    for(int x = 0; x < src.cols(); x++) {
        padded.at<double>(y + 2, x + 2) = src.getMat().at<uchar>(y, x);
    }
}
// 卷积操作...
result.convertTo(dst, CV_8U);
```

---

### 3. Canny边缘检测的实现

#### 3.1 算法原理

Canny边缘检测是一种多级边缘检测算法，主要包含以下步骤：

1. **高斯滤波**：平滑图像，减少噪声
2. **计算梯度**：使用Sobel算子计算图像梯度
3. **非极大值抑制（NMS）**：细化边缘
4. **双阈值处理**：区分强边缘和弱边缘
5. **滞后边界跟踪**：连接弱边缘到强边缘

#### 3.2 Sobel梯度计算

使用Sobel算子计算x和y方向的梯度：

```cpp
cv::Mat sobelX = (cv::Mat_<double>(3, 3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);
cv::Mat sobelY = (cv::Mat_<double>(3, 3) <<
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1);

// 计算梯度幅值和方向
double magnitude_value = sqrt(sumX * sumX + sumY * sumY);
double direction = atan2(sumY, sumX) * 180.0 / CV_PI;
```

#### 3.3 非极大值抑制

将梯度方向量化到4个方向（0°、45°、90°、135°），沿梯度方向比较当前像素与相邻像素，只保留局部最大值：

```cpp
if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
    neighbor1 = magMat.at<double>(y, x + 1);  // 水平右
    neighbor2 = magMat.at<double>(y, x - 1);  // 水平左
}
// ... 其他方向类似

if(mag < neighbor1 || mag < neighbor2) {
    result.at<double>(y, x) = 0.0;  // 非极大值抑制
}
```

#### 3.4 双阈值处理

设置高阈值和低阈值，将边缘分为强边缘、弱边缘和非边缘：

```cpp
if(val >= highThreshold) {
    edgesMat.at<uchar>(y, x) = 255;  // 强边缘
} else if(val >= lowThreshold) {
    edgesMat.at<uchar>(y, x) = 128;  // 弱边缘
} else {
    edgesMat.at<uchar>(y, x) = 0;    // 非边缘
}
```

---

### 4. 霍夫圆变换的实现

#### 4.1 算法原理

霍夫圆变换的基本思想是将图像空间中的圆映射到参数空间中的点。圆的方程为：

$$(x-a)^2 + (y-b)^2 = r^2$$

其中(a, b)是圆心坐标，r是半径。在参数空间中，每个边缘点会对所有可能的圆心位置进行投票。

#### 4.2 实现代码

```cpp
auto inline m_houghCircle(cv::InputArray src, cv::OutputArray circles,
                          double dp, double minDist, double param1,
                          double param2, int minRadius, int maxRadius) -> void {
    // 1. 收集所有边缘点
    std::vector<cv::Point> edgePoints;
    for(int y = 0; y < edgeMat.rows; y++) {
        for(int x = 0; x < edgeMat.cols; x++) {
            if(edgeMat.at<uchar>(y, x) > 0) {
                edgePoints.push_back(cv::Point(x, y));
            }
        }
    }

    // 2. 对每个可能的半径开始投票
    for (int r = minRadius; r <= maxRadius; r++) {
        cv::Mat acc = cv::Mat::zeros(rows, cols, CV_32S);

        // 预计算圆周上的位移点
        std::vector<cv::Point> offsets;
        int steps = std::max(10, static_cast<int>(2 * CV_PI * r_dp));
        for (int s = 0; s < steps; s++) {
            double theta = 2.0 * CV_PI * s / steps;
            int dx = std::round(r_dp * cos(theta));
            int dy = std::round(r_dp * sin(theta));
            offsets.push_back(cv::Point(dx, dy));
        }

        // 对边缘点投票
        for(const auto& pt : edgePoints) {
            int a = x_dp - off.x;
            int b = y_dp - off.y;
            if(a >= 0 && a < cols && b >= 0 && b < rows) {
                acc.at<int>(b, a)++;
            }
        }
    }

    // 3. 提取票数超过阈值的候选圆
    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {
            if(acc.at<int>(y, x) >= param2) {
                allCandidates.push_back({...});
            }
        }
    }

    // 4. 非极大值抑制，去除距离过近的候选圆
    std::sort(allCandidates.begin(), allCandidates.end(), ...);
    for(const auto& cand : allCandidates) {
        bool keep = true;
        for(const auto& fc : finalCircles) {
            double dist = sqrt(pow(cand.center.x - fc[0], 2) + ...);
            if(dist < minDist) {
                keep = false;
                break;
            }
        }
        if(keep) {
            finalCircles.push_back(...);
        }
    }
}
```

---

### 5. 四种实现对比

为了对比不同实现的效果，我们构建了四种组合：

| 方法 | 高斯模糊 | 边缘检测 | 霍夫圆变换 |
|------|---------|---------|-----------|
| 方法1 | OpenCV | OpenCV | OpenCV |
| 方法2 | 我们的 | OpenCV | OpenCV |
| 方法3 | 我们的 | 我们的 | OpenCV |
| 方法4 | 我们的 | 我们的 | 我们的 |

---

## 实验结果

### 检测输出
```
========================================
钱币定位系统 - 四种实现对比
========================================
========================================
方法1：全部使用OpenCV
检测到 6 个钱币
  钱币 1: 圆心(1552, 1214), 半径 152
  钱币 2: 圆心(648, 618), 半径 129
  钱币 3: 圆心(2138, 1928), 半径 121
  钱币 4: 圆心(936, 1798), 半径 123
  钱币 5: 圆心(2894, 1338), 半径 131
  钱币 6: 圆心(2476, 638), 半径 121
耗时: 507.788 ms
========================================
方法2：使用我们的高斯模糊 + OpenCV边缘检测 + OpenCV霍夫圆
检测到 6 个钱币
  钱币 1: 圆心(1552, 1214), 半径 152
  钱币 2: 圆心(648, 618), 半径 129
  钱币 3: 圆心(2138, 1924), 半径 121
  钱币 4: 圆心(930, 1806), 半径 122
  钱币 5: 圆心(2894, 1338), 半径 131
  钱币 6: 圆心(2474, 638), 半径 119
耗时: 784.332 ms
========================================
方法3：使用我们的高斯模糊 + 我们的边缘检测 + OpenCV霍夫圆
检测到 4 个钱币
  钱币 1: 圆心(1550, 1214), 半径 155
  钱币 2: 圆心(930, 1806), 半径 123
  钱币 3: 圆心(648, 620), 半径 128
  钱币 4: 圆心(2140, 1926), 半径 124
耗时: 1187.16 ms
========================================
方法4：全部使用我们的实现
检测到 19 个钱币
  钱币 1: 圆心(1552, 1217), 半径 152
  钱币 2: 圆心(649, 621), 半径 130
  钱币 3: 圆心(930, 1806), 半径 121
  钱币 4: 圆心(2144, 1923), 半径 129
  钱币 5: 圆心(2474, 639), 半径 120
  钱币 6: 圆心(2888, 1340), 半径 131
  钱币 7: 圆心(3257, 199), 半径 198
  钱币 8: 圆心(199, 199), 半径 198
  钱币 9: 圆心(3257, 2349), 半径 198
  钱币 10: 圆心(199, 2349), 半径 198
  钱币 11: 圆心(1208, 1209), 半径 199
  钱币 12: 圆心(1250, 1799), 半径 199
  钱币 13: 圆心(1871, 1224), 半径 185
  钱币 14: 圆心(1552, 1569), 半径 200
  钱币 15: 圆心(3207, 1380), 半径 191
  钱币 16: 圆心(1826, 1894), 半径 198
  钱币 17: 圆心(1520, 869), 半径 195
  钱币 18: 圆心(328, 626), 半径 197
  钱币 19: 圆心(965, 1484), 半径 199
耗时: 4998.56 ms
========================================
耗时对比总结
========================================
方法1 (OpenCV全部):     507.788 ms
方法2 (我们的模糊):      784.332 ms
方法3 (我们的模糊+边缘): 1187.16 ms
方法4 (全部我们的):      4998.56 ms
总耗时: 7613.02 ms

结果已保存到 /home/kazusa/computer_vision/lab02/output/
```

### 边缘检测效果

下图展示了我们实现的Canny边缘检测的效果：

![边缘检测结果](../output/edge.jpg)

### 圆形检测结果

下图展示了钱币检测的结果（绿色圆圈为检测到的钱币，红色点为圆心）：

![检测结果](../output/result.jpg)

### 检测参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| dp | 1 | 累加器分辨率 |
| minDist | rows/8 | 圆心之间的最小距离 |
| param1 | 100 | Canny边缘检测的高阈值 |
| param2 | 30 | 霍夫圆变换的投票阈值 |
| minRadius | 80 | 最小半径 |
| maxRadius | 200 | 最大半径 |

---

## 实验小结

### 1. 高斯滤波的实现

通过手动实现高斯滤波，我们理解了：
- 二维高斯函数的权重分布特性
- 卷积运算的基本原理
- 边缘填充方式对结果的影响

### 2. Canny边缘检测的实现

自主实现Canny算法让我们深入理解了：
- Sobel算子如何计算梯度
- 非极大值抑制如何细化边缘
- 双阈值检测如何区分强边缘和弱边缘
- 滞后跟踪如何连接断裂的边缘

### 3. 霍夫圆变换的实现

通过实现霍夫圆变换，我们掌握了：
- 从图像空间到参数空间的映射原理
- 累加器投票机制
- 非极大值抑制在圆检测中的应用

### 4. 与OpenCV实现的对比

我们的实现与OpenCV相比：
- 功能上基本一致，能够检测出图像中的钱币
- 由于未使用霍夫梯度法优化，检测到了更多的"虚空圆"
- 手动实现加深了对算法原理的理解

### 5. 进一步优化的方向

当前实现可以进一步优化：
- 使用霍夫梯度法替代标准霍夫变换，减少计算量
- 使用DFS（深度优先搜索）进行边缘跟踪，更好地保留弱边缘
- 优化内存使用，减少累加器的空间开销
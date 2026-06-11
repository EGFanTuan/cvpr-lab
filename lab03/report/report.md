# 计算机视觉与模式识别实验报告

## 实验目的
- 理解基于2D特征的图像配准方法的基本原理；
- 掌握SIFT（Scale-Invariant Feature Transform）特征检测与匹配算法；
- 熟悉OpenCV Stitcher模块的全景图拼接流程；
- 通过对比不同输入组的拼接效果，理解影响全景图拼接质量的关键因素。

## 实验内容
- 基于多幅图像（2幅及以上），运用基于2D特征的配准方法，制作全景图。
- 使用OpenCV的SIFT特征检测器和Stitcher拼接模块完成系统设计。
- 设置多组输入进行对比实验：
  - **第一组**：4幅图像（pic_01~04），验证多图拼接能力；
  - **第二组**（对照）：4幅图像（pic_05~08），测试重叠度不足时的拼接效果；
  - **第三组**：2幅图像（pic_09~10），测试双图拼接；
  - **第四组**：2幅图像（pic_11~12），测试双图拼接。

## 实验过程

### 1. 算法整体流程

全景图拼接系统的主要流程如下：

```
输入图像组 → SIFT特征检测 → 特征匹配 → 单应性矩阵估计 → 图像变形 → 曝光补偿 → 图像融合 → 输出全景图
```

OpenCV的`cv::Stitcher`模块封装了完整的拼接管线，各阶段功能如下：

| 阶段 | 功能 | 使用的算法/技术 |
|------|------|----------------|
| 特征检测 | 在每幅图像中提取关键点和描述子 | SIFT |
| 特征匹配 | 在图像对之间建立特征点对应关系 | kNN匹配 + Lowe's ratio test |
| 相机参数估计 | 估计每幅图像的相机内参和旋转矩阵 | RANSAC + 光束平差法 |
| 图像变形 | 将各图像投影到统一的全景平面 | 球面/柱面/平面投影 |
| 曝光补偿 | 消除图像间的亮度差异 | 增益补偿 |
| 图像融合 | 无缝拼接各图像 | 多频段融合（Multi-band blending） |

---

### 2. SIFT特征检测

#### 2.1 算法原理

SIFT（Scale-Invariant Feature Transform）由David Lowe于1999年提出，是一种对图像缩放、旋转、仿射变换和光照变化均具有不变性的局部特征描述算法。其主要步骤包括：

1. **尺度空间极值检测**：通过不同尺度的高斯差分（DoG）金字塔检测潜在的兴趣点

   $$D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y)$$

2. **关键点精确定位**：对DoG空间的极值点进行亚像素级精确定位，剔除低对比度和边缘响应点

3. **方向分配**：基于局部梯度方向直方图，为每个关键点分配一个或多个主方向

   $$m(x, y) = \sqrt{(L(x+1, y) - L(x-1, y))^2 + (L(x, y+1) - L(x, y-1))^2}$$

   $$\theta(x, y) = \tan^{-1}\left(\frac{L(x, y+1) - L(x, y-1)}{L(x+1, y) - L(x-1, y)}\right)$$

4. **关键点描述子**：在关键点周围4×4子区域内计算8方向梯度直方图，形成128维特征向量

#### 2.2 在OpenCV中的使用

```cpp
cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
// SIFT特征检测已集成在Stitcher管线中，无需显式调用
```

---

### 3. 特征匹配与几何验证

#### 3.1 kNN匹配与Lowe's Ratio Test

对于两幅图像的特征描述子，使用k近邻（k=2）搜索找到每个特征点的最近邻和次近邻匹配。通过Lowe's ratio test过滤错误匹配：当最近邻距离与次近邻距离之比小于阈值（通常为0.7~0.8）时，保留该匹配。

$$\frac{d_{nearest}}{d_{second\_nearest}} < 0.75$$

#### 3.2 RANSAC与单应性矩阵估计

使用RANSAC（Random Sample Consensus）算法鲁棒地估计图像间的单应性矩阵$H$：

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad H = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix}$$

RANSAC通过迭代随机采样最小点集（4对匹配点）估计$H$，并选择内点数量最多的模型，有效剔除外点（错误匹配）的影响。

---

### 4. 图像融合

#### 4.1 曝光补偿

由于拍摄角度和光照条件不同，相邻图像间可能存在亮度差异。曝光补偿通过为每幅图像估计一个增益系数来均衡整体亮度：

$$I_i'(x, y) = g_i \cdot I_i(x, y)$$

#### 4.2 多频段融合

多频段融合（Multi-band Blending）将图像分解为不同频率的拉普拉斯金字塔，在不同频段上以不同宽度进行加权融合，从而避免拼接缝处的明显过渡。

---

### 5. 实验设计与分组

为了验证不同条件下的拼接效果，设计了四组对比实验：

| 组别 | 输入图像 | 图像数量 | 状态 | 说明 |
|------|---------|---------|------|------|
| 第一组 | pic_01~04 | 4幅 | ✅ 启用 | 测试多图拼接能力 |
| 第二组 | pic_05~08 | 4幅 | ❌ 对照 | 重叠度不足，预期失败 |
| 第三组 | pic_09~10 | 2幅 | ✅ 启用 | 测试双图拼接 |
| 第四组 | pic_11~12 | 2幅 | ✅ 启用 | 测试双图拼接 |

#### 实现代码

```cpp
// Define groups of input images
std::vector<std::vector<std::string>> imageGroups = {
    {
      input_path + "pic_01.jpg",
      input_path + "pic_02.jpg",
      input_path + "pic_03.jpg",
      input_path + "pic_04.jpg",
    },
    // {   // 第二组：作为对照，已注释
    //   input_path + "pic_05.jpg",
    //   input_path + "pic_06.jpg",
    //   input_path + "pic_07.jpg",
    //   input_path + "pic_08.jpg",
    // },
    {
      input_path + "pic_09.jpg",
      input_path + "pic_10.jpg",
    },
    {
      input_path + "pic_11.jpg",
      input_path + "pic_12.jpg",
    }
};

// Process each group
for (size_t g = 0; g < imageGroups.size(); ++g) {
    std::vector<cv::Mat> mats;
    for (const auto& path : imageGroups[g]) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "Failed to read image: " << path << std::endl;
            return -1;
        }
        mats.push_back(img);
    }

    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);

    cv::Mat pano;
    cv::Stitcher::Status status = stitcher->stitch(mats, pano);
    if (status != cv::Stitcher::OK) {
        std::cerr << "Error during stitching group " << (g + 1)
                  << ", error code: " << int(status) << std::endl;
    }

    std::string output_path = outputDir + "panorama_" + std::to_string(g + 1) + ".jpg";
    cv::imwrite(output_path, pano);
}
```

---

## 实验结果

### 第一组：4幅图像拼接（pic_01 ~ pic_04）

拼接成功。4幅图像被正确配准并融合为一张完整的全景图。

![全景图1](../output/panorama_1.jpg)

### 第二组：4幅图像拼接（pic_05 ~ pic_08，对照）

拼接失败，返回错误码 `ERR_NEED_MORE_IMGS (3)`，表示无法在图像之间找到足够多的匹配特征点来估计相机参数。

**失败原因分析**：
- 该组图像之间的重叠区域不足，或图像内容纹理较弱，导致SIFT检测到的特征点过少；
- 特征匹配阶段无法建立足够数量的可靠对应关系，RANSAC无法估计有效的单应性矩阵；
- 拼接器在相机参数估计阶段失败，无法确定各图像的相对位置关系。

该组作为**对照实验**保留在代码中（已注释），用于说明全景图拼接的前提条件：**图像之间必须有足够的重叠区域（通常建议30%~50%）**。

### 第三组：2幅图像拼接（pic_09 ~ pic_10）

拼接成功。两幅图像被配准并融合为全景图。

![全景图2](../output/panorama_2.jpg)

### 第四组：2幅图像拼接（pic_11 ~ pic_12）

拼接成功。两幅图像被配准并融合为全景图。

![全景图3](../output/panorama_3.jpg)

### 拼接错误码速查表

| 错误码 | 名称 | 含义 |
|--------|------|------|
| 0 | OK | 拼接成功 |
| 1 | ERR_NEED_MORE_IMGS | 需要更多图像或匹配点不足 |
| 2 | ERR_HOMOGRAPHY_EST_FAIL | 单应性矩阵估计失败 |
| 3 | ERR_CAMERA_PARAMS_ADJUST_FAIL | 相机参数调整失败 |

---

## 实验小结

### 1. 全景图拼接的核心要素

通过本实验，全景图拼接的关键在于：
- **足够的重叠区域**：相邻图像之间必须共享足够的场景内容，才能建立可靠的特征匹配；
- **丰富的纹理特征**：SIFT依赖图像的局部梯度信息，纹理贫乏的区域（如蓝天、白墙）难以提取有效特征点；
- **合理的拍摄策略**：拍摄时应保持相机近似绕光心旋转，减少视差带来的配准误差。

### 2. SIFT特征的优势

SIFT特征在本实验中展现了良好的性能：
- **尺度不变性**：能够匹配不同距离拍摄的相同场景；
- **旋转不变性**：对手持拍摄的角度偏差具有鲁棒性；
- **光照不变性**：对曝光差异具有一定容忍度。

### 3. 第二组失败的经验教训

第二组（pic_05~08）的拼接失败印证了**图像重叠度对拼接成功率的关键影响**。当Stitcher返回`ERR_NEED_MORE_IMGS`时，说明图像间无法建立足够的特征对应关系。解决方案包括：
- 拍摄时确保相邻帧之间有30%~50%的重叠；
- 适当增加图像数量以提高覆盖密度；
- 对于低纹理场景，可考虑使用结构化光或人工标记辅助配准。

### 4. 双图 vs 多图拼接

第一组使用4幅图像进行拼接，相比第三、四组的2幅图像拼接，覆盖了更大的视野范围。多图拼接的挑战在于：
- 累积误差会随图像数量增加而增大；
- 全局光束平差法（Bundle Adjustment）有助于减少累积漂移；
- OpenCV的Stitcher模块已内置了全局优化机制，能够较好地处理多图场景。

### 5. 进一步优化的方向

- 尝试不同的特征检测器（如AKAZE、ORB）进行对比，分析各算法在全景拼接中的适用性；
- 调整拼接参数（如置信度阈值`pano_confidence_thresh`、融合宽度等）以适应不同场景；
- 使用柱面投影或球面投影替代默认的平面投影，以提高大视角场景的拼接质量；
- 探索基于深度学习的特征匹配方法（如SuperPoint + SuperGlue），在挑战性场景中获得更好的匹配效果。

### 6. 仓库地址
- GitHub: [https://github.com/EGFanTuan/cvpr-lab]

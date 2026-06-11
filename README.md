# Computer Vision Labs

计算机视觉与模式识别课程实验仓库，包含三个实验项目。

## 实验内容

### Lab01 - 元素图像分析
统计图像中 Al、Fe、P 三种化学元素的数目，分析它们之间的两两重叠情况以及三者共同重叠的情况。

### Lab02 - 钱币定位系统
使用 Canny 边缘检测和 Hough 圆检测算法，实现图像中钱币的定位，检测钱币的圆心坐标与半径。同时自主实现了高斯滤波、Canny边缘检测和霍夫圆变换，与OpenCV实现进行对比。

### Lab03 - 全景图拼接
基于SIFT特征检测与匹配，使用OpenCV Stitcher模块实现多幅图像的全景图拼接。设置多组输入进行对比实验，分析图像重叠度对拼接成功率的影响。

## 环境依赖

- CMake >= 3.10
- C++17
- OpenCV 4.x

## 构建指南

### 构建所有实验

```bash
mkdir build
cd build
cmake ..
make
```

### 分别构建某个实验

```bash
# Lab01
mkdir build/lab01
cd build/lab01
cmake ../../lab01
make

# Lab02
mkdir build/lab02
cd build/lab02
cmake ../../lab02
make
```

## 运行实验

```bash
# Lab01
./build/lab01/lab01

# Lab02
./build/lab02/lab02

# Lab03
./build/lab03/lab03
```

## 项目结构

```
computer_vision/
├── CMakeLists.txt          # 顶层 CMake 配置
├── lab01/                  # 实验一：元素图像分析
│   ├── src/                # 源代码
│   ├── input/              # 输入图像
│   ├── output/             # 输出结果
│   └── report/             # 实验报告
├── lab02/                  # 实验二：钱币定位系统
│   ├── src/                # 源代码
│   ├── input/              # 输入图像
│   ├── output/             # 输出结果
│   └── report/             # 实验报告
├── lab03/                  # 实验三：全景图拼接
│   ├── src/                # 源代码
│   ├── input/              # 输入图像
│   ├── output/             # 输出全景图
│   └── report/             # 实验报告
└── build/                  # 构建目录
```

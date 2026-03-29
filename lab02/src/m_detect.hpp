#pragma once
#include <opencv2/opencv.hpp>

auto static m_CreateGaussianKernel(int ksize, double sigma) -> cv::Mat {
  cv::Mat kernel(ksize, ksize, CV_64F);
  int halfSize = ksize / 2;
  double sum = 0.0;

  for (int y = -halfSize; y <= halfSize; y++) {
    for (int x = -halfSize; x <= halfSize; x++) {
      // 二维高斯核函数
      double g = 
      (1 / (2 * CV_PI * sigma * sigma)) * 
      exp(-(x * x + y * y) / (2 * sigma * sigma));
      kernel.at<double>(y + halfSize, x + halfSize) = g;
      sum += g;
    }
  }

  // 归一化核，使其元素和为1
  for(int y = 0; y < ksize; y++) {
    for(int x = 0; x < ksize; x++) {
      kernel.at<double>(y, x) /= sum;
    }
  }

  return kernel;
}

auto static m_CalculateGradientWithSobel(cv::InputArray src, 
                                         cv::OutputArray gradX, 
                                         cv::OutputArray gradY, 
                                         cv::OutputArray magnitude, 
                                         cv::OutputArray direction) -> void {
  cv::Mat srcMat = src.getMat();
  
  // 初始化Sobel算子
  cv::Mat sobelX = (cv::Mat_<double>(3, 3) << 
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);
  cv::Mat sobelY = (cv::Mat_<double>(3, 3) << 
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1);

  cv::Mat m_gradX(src.size(), CV_64F);
  cv::Mat m_gradY(src.size(), CV_64F);
  cv::Mat m_magnitude(src.size(), CV_64F);
  cv::Mat m_direction(src.size(), CV_64F);

  // 对每个像素应用Sobel算子，计算梯度
  for(int y = 1; y < srcMat.rows - 1; y++) {
    for(int x = 1; x < srcMat.cols - 1; x++) {
      double sumX = 0.0, sumY = 0.0;
      for(int j = -1; j <= 1; j++) {
        for(int i = -1; i <= 1; i++) {
          double pixelVal = srcMat.at<uchar>(y + j, x + i);
          sumX += pixelVal * sobelX.at<double>(j + 1, i + 1);
          sumY += pixelVal * sobelY.at<double>(j + 1, i + 1);
        }
      }
      m_gradX.at<double>(y, x) = sumX;
      m_gradY.at<double>(y, x) = sumY;

      // 计算幅值
      double magnitude_value = sqrt(sumX * sumX + sumY * sumY);
      m_magnitude.at<double>(y, x) = magnitude_value;
      // 计算方向，使用 atan2 来获得正确的象限
      if(magnitude_value > 0) {
        m_direction.at<double>(y, x) = atan2(sumY, sumX) * 180.0 / CV_PI; // 转换为角度
      }
      else {
        m_direction.at<double>(y, x) = 0.0; // 无梯度时方向设为0
      }
    }
  }

  // 输出结果 - 直接赋值给OutputArray
  gradX.assign(m_gradX);
  gradY.assign(m_gradY);
  magnitude.assign(m_magnitude);
  direction.assign(m_direction);
}

auto static m_NMSCalculate(cv::InputArray magnitude, cv::InputArray direction, cv::OutputArray nms) -> void {
  cv::Mat magMat = magnitude.getMat();
  cv::Mat dirMat = direction.getMat();
  
  cv::Mat result = magMat.clone(); // 复制原始幅值作为结果
  
  auto H = magMat.rows, W = magMat.cols;
  for(int y = 1; y < H - 1; y++) {
    for(int x = 1; x < W - 1; x++) {
      double angle = dirMat.at<double>(y, x);

      // 将角度转换为0-180范围
      if (angle < 0) angle += 180;

      double mag = magMat.at<double>(y, x);
      double neighbor1 = 0.0, neighbor2 = 0.0;
      // 根据方向选择比较的邻居
      if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
        neighbor1 = magMat.at<double>(y, x + 1); // 水平右
        neighbor2 = magMat.at<double>(y, x - 1); // 水平左
      }
      else if(angle >= 22.5 && angle < 67.5) {
        neighbor1 = magMat.at<double>(y - 1, x + 1); // 右上
        neighbor2 = magMat.at<double>(y + 1, x - 1); // 左下
      }
      else if(angle >= 67.5 && angle < 112.5) {
        neighbor1 = magMat.at<double>(y - 1, x); // 垂直上
        neighbor2 = magMat.at<double>(y + 1, x); // 垂直下
      }
      else if(angle >= 112.5 && angle < 157.5) {
        neighbor1 = magMat.at<double>(y - 1, x - 1); // 左上
        neighbor2 = magMat.at<double>(y + 1, x + 1); // 右下
      }

      // 非极大值抑制
      if(mag < neighbor1 || mag < neighbor2) {
        result.at<double>(y, x) = 0.0; // 抑制非极大值
      }
    }
  }

  nms.assign(result);
}

auto inline m_edgeDetect(cv::InputArray src, cv::OutputArray edges, double lowThreshold, double highThreshold) -> void {
  cv::Mat gradX, gradY, magnitude, direction;
  m_CalculateGradientWithSobel(src, gradX, gradY, magnitude, direction);

  cv::Mat nms8u;
  m_NMSCalculate(magnitude, direction, nms8u);

  // 为输出数组分配内存
  edges.create(src.size(), CV_8U);
  cv::Mat edgesMat = edges.getMat();

  // 双阈值处理
  for(int y = 0; y < nms8u.rows; y++) {
    for(int x = 0; x < nms8u.cols; x++) {
      double val = nms8u.at<double>(y, x);
      if(val >= highThreshold) {
        edgesMat.at<uchar>(y, x) = 255; // 强边缘
      }
      else if(val >= lowThreshold) {
        edgesMat.at<uchar>(y, x) = 128; // 弱边缘
      }
      else {
        edgesMat.at<uchar>(y, x) = 0; // 非边缘
      }
    }
  }

  // 连接边缘（简单的8连通检查）
  for(int y = 1; y < nms8u.rows - 1; y++) {
    for(int x = 1; x < nms8u.cols - 1; x++) {
      if(edgesMat.at<uchar>(y, x) == 128) {
        // 检查8邻域是否有强边缘
        bool connectedToStrong = false;
        for(int j = -1; j <= 1; j++) {
          for(int i = -1; i <= 1; i++) {
            if(edgesMat.at<uchar>(y + j, x + i) == 255) {
              connectedToStrong = true;
              break;
            }
          }
        }
        if(connectedToStrong) {
          edgesMat.at<uchar>(y, x) = 255; // 连接到强边缘
        }
        else {
          edgesMat.at<uchar>(y, x) = 0; // 否则抑制
        }
      }
    }
  }
}

/**
 * @brief 对输入图像进行高斯模糊处理，使用一个5*5的高斯核，标准差为1.5。
 */
auto inline m_GaussianBlur(cv::InputArray src, cv::OutputArray dst, int ksize = 5, double sigma = 1.5) -> void {
  auto kernel = m_CreateGaussianKernel(ksize, sigma);

  // 填充边缘
  // 可以换成更好的填充方式
  cv::Mat padded = cv::Mat::zeros(src.rows() + 4, src.cols() + 4, CV_64F);
  for(int y = 0; y < src.rows(); y++) {
    for(int x = 0; x < src.cols(); x++) {
      padded.at<double>(y + 2, x + 2) = src.getMat().at<uchar>(y, x);
    }
  }

  auto H = src.rows(), W = src.cols();
  cv::Mat result = cv::Mat::zeros(H, W, CV_64F);

  // 卷积操作
  for(int y = 0; y < H; y++) {
    for(int x = 0; x < W; x++) {
      double sum = 0.0;
      for(int j = 0; j < ksize; j++) {
        for(int i = 0; i < ksize; i++) {
          sum += padded.at<double>(y + j, x + i) * kernel.at<double>(j, i);
        }
      }
      result.at<double>(y, x) = sum;
    }
  }

  // 转换回8位图像
  result.convertTo(dst, CV_8U);
}
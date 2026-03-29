#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include "m_detect.hpp"

using namespace cv;
using namespace std;

void drawCircles(Mat& image, const vector<Vec3f>& circles, const string& prefix) {
    Mat result = image.clone();
    for (size_t i = 0; i < circles.size(); i++) {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        int radius = c[2];
        circle(result, center, radius, Scalar(0, 255, 0), 2);
        circle(result, center, 3, Scalar(0, 0, 255), -1);
    }
    imwrite(prefix + "_result.jpg", result);
}

void printCircles(const vector<Vec3f>& circles, const string& method) {
    cout << "========================================" << endl;
    cout << method << endl;
    cout << "检测到 " << circles.size() << " 个钱币" << endl;
    for (size_t i = 0; i < circles.size(); i++) {
        Vec3i c = circles[i];
        cout << "  钱币 " << (i + 1) << ": 圆心(" << c[0] << ", " << c[1] << "), 半径 " << c[2] << endl;
    }
}

int main(int argc, char** argv) {
    string imagePath = string(PROJECT_DIR) + "/input/picture.jpg";
    if (argc > 1) {
        imagePath = argv[1];
    }

    Mat image = imread(imagePath);
    if (image.empty()) {
        cerr << "无法读取图片: " << imagePath << endl;
        return -1;
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    string outputDir = string(PROJECT_DIR) + "/output/";
    std::error_code ec;
    std::filesystem::create_directories(outputDir, ec);
    if (ec) {
        cerr << "创建输出目录失败: " << outputDir << "，错误: " << ec.message() << endl;
        return -1;
    }

    vector<Vec3f> circles;
    Mat edgeImage, blurImage;
    double minDist = gray.rows / 8.0;
    int lowThresh = 30, highThresh = 150;
    int minRadius = 80, maxRadius = 200;

    cout << "========================================" << endl;
    cout << "钱币定位系统 - 四种实现对比" << endl;
    cout << "========================================" << endl;

    auto startTotal = chrono::high_resolution_clock::now();

    auto start = chrono::high_resolution_clock::now();
    GaussianBlur(gray, blurImage, Size(5, 5), 1.5);
    Canny(blurImage, edgeImage, lowThresh, highThresh);
    HoughCircles(edgeImage, circles, HOUGH_GRADIENT, 1, minDist, 100, 30, minRadius, maxRadius);
    auto end = chrono::high_resolution_clock::now();
    double time1 = chrono::duration<double, milli>(end - start).count();
    printCircles(circles, "方法1：全部使用OpenCV");
    cout << "耗时: " << time1 << " ms" << endl;
    drawCircles(image, circles, outputDir + "method1_opencv_all");

    start = chrono::high_resolution_clock::now();
    m_GaussianBlur(gray, blurImage, 5, 1.5);
    Canny(blurImage, edgeImage, lowThresh, highThresh);
    HoughCircles(edgeImage, circles, HOUGH_GRADIENT, 1, minDist, 100, 30, minRadius, maxRadius);
    end = chrono::high_resolution_clock::now();
    double time2 = chrono::duration<double, milli>(end - start).count();
    printCircles(circles, "方法2：使用我们的高斯模糊 + OpenCV边缘检测 + OpenCV霍夫圆");
    cout << "耗时: " << time2 << " ms" << endl;
    drawCircles(image, circles, outputDir + "method2_our_blur");

    start = chrono::high_resolution_clock::now();
    m_GaussianBlur(gray, blurImage, 5, 1.5);
    m_edgeDetect(blurImage, edgeImage, lowThresh, highThresh);
    HoughCircles(edgeImage, circles, HOUGH_GRADIENT, 1, minDist, 100, 30, minRadius, maxRadius);
    end = chrono::high_resolution_clock::now();
    double time3 = chrono::duration<double, milli>(end - start).count();
    printCircles(circles, "方法3：使用我们的高斯模糊 + 我们的边缘检测 + OpenCV霍夫圆");
    cout << "耗时: " << time3 << " ms" << endl;
    drawCircles(image, circles, outputDir + "method3_our_blur_edge");

    start = chrono::high_resolution_clock::now();
    m_GaussianBlur(gray, blurImage, 5, 1.5);
    m_edgeDetect(blurImage, edgeImage, lowThresh, highThresh);
    m_houghCircle(edgeImage, circles, 1, minDist, 100, 30, minRadius, maxRadius);
    end = chrono::high_resolution_clock::now();
    double time4 = chrono::duration<double, milli>(end - start).count();
    printCircles(circles, "方法4：全部使用我们的实现");
    cout << "耗时: " << time4 << " ms" << endl;
    drawCircles(image, circles, outputDir + "method4_our_all");

    auto endTotal = chrono::high_resolution_clock::now();
    double timeTotal = chrono::duration<double, milli>(endTotal - startTotal).count();

    cout << "========================================" << endl;
    cout << "耗时对比总结" << endl;
    cout << "========================================" << endl;
    cout << "方法1 (OpenCV全部):     " << time1 << " ms" << endl;
    cout << "方法2 (我们的模糊):      " << time2 << " ms" << endl;
    cout << "方法3 (我们的模糊+边缘): " << time3 << " ms" << endl;
    cout << "方法4 (全部我们的):      " << time4 << " ms" << endl;
    cout << "总耗时: " << timeTotal << " ms" << endl;
    cout << "\n结果已保存到 " << outputDir << endl;

    return 0;
}
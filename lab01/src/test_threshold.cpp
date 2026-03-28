#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int countElements(const Mat& mask) {
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    int count = 0;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > 2) {
            count++;
        }
    }
    return count;
}

int main() {
    string inputPath = "/home/kazusa/computer_vision/lab01/input/Fe.jpg";
    string outputPath = "/home/kazusa/computer_vision/lab01/output/";
    
    Mat image = imread(inputPath);
    if (image.empty()) {
        cout << "无法读取图像: " << inputPath << endl;
        return -1;
    }
    
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    
    cout << "V值阈值对比测试 (P.jpg)" << endl;
    cout << "========================" << endl;
    
    for (int v = 0; v <= 100; v += 10) {
        Mat mask;
        inRange(hsv, Scalar(0, 0, v), Scalar(180, 255, 255), mask);
        
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);
        
        int count = countElements(mask);
        cout << "V > " << v << " : " << count << " 个元素" << endl;
        
        string outPath = outputPath + "P_threshold_V" + to_string(v) + ".jpg";
        imwrite(outPath, mask);
    }
    
    cout << "\n所有对比图像已保存到 output 目录" << endl;
    return 0;
}

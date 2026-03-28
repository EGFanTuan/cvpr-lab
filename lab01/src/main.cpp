#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// 函数声明
Mat preprocessImage(const Mat& image);
int countElements(const Mat& image);
Mat createBinaryMask(const Mat& image);
void saveOverlapImage(const Mat& mask1, const Mat& mask2, const string& outputPath, const Scalar& color);
void saveTripleOverlapImage(const Mat& mask1, const Mat& mask2, const Mat& mask3, const string& outputPath);

int main() {
    // 图像路径
    string inputPath = "/home/kazusa/computer_vision/lab01/input/";
    string outputPath = "/home/kazusa/computer_vision/lab01/output/";
    vector<string> elements = {"Al", "Fe", "P"};
    vector<Mat> masks;
    vector<int> counts;
    
    // 处理每个元素的图像
    for (const string& elem : elements) {
        string imagePath = inputPath + elem + ".jpg";
        Mat image = imread(imagePath);
        if (image.empty()) {
            cout << "无法读取图像: " << imagePath << endl;
            return -1;
        }
        
        // 创建二值掩码
        Mat mask = createBinaryMask(image);
        masks.push_back(mask);
        
        // 保存二值图
        string binaryPath = outputPath + elem + "_binary.jpg";
        imwrite(binaryPath, mask);
        
        // 统计元素数目
        int count = countElements(mask);
        counts.push_back(count);
        cout << elem << " 元素数目: " << count << endl;
    }
    
    // 计算两两元素重叠数目并生成重叠图像
    vector<pair<int, int>> pairs = {{0, 1}, {1, 2}, {0, 2}};
    vector<string> pairNames = {"Al_Fe", "Fe_P", "Al_P"};
    vector<Scalar> pairColors = {Scalar(0, 255, 255), Scalar(255, 0, 255), Scalar(255, 255, 0)};
    
    for (int i = 0; i < pairs.size(); i++) {
        int idx1 = pairs[i].first;
        int idx2 = pairs[i].second;
        Mat overlap = masks[idx1] & masks[idx2];
        int overlapCount = countElements(overlap);
        cout << elements[idx1] << " 和 " << elements[idx2] << " 重叠数目: " << overlapCount << endl;
        
        // 保存重叠图像
        string outPath = outputPath + pairNames[i] + "_overlap.jpg";
        saveOverlapImage(masks[idx1], masks[idx2], outPath, pairColors[i]);
    }
    
    // 计算三种元素重叠数目并生成重叠图像
    Mat tripleOverlap = masks[0] & masks[1] & masks[2];
    int tripleCount = countElements(tripleOverlap);
    cout << "Al、Fe 和 P 三者重叠数目: " << tripleCount << endl;
    
    // 保存三种元素重叠图像
    string tripleOutPath = outputPath + "Al_Fe_P_overlap.jpg";
    saveTripleOverlapImage(masks[0], masks[1], masks[2], tripleOutPath);
    
    cout << "实验完成，结果已保存到 output 目录" << endl;
    return 0;
}

// 创建二值掩码
Mat createBinaryMask(const Mat& image) {
    Mat hsv, mask;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    
    // 由于不同元素颜色不同，我们使用一个较宽的阈值范围来捕获所有非黑色区域
    // 假设元素颜色不是黑色，背景是黑色
    inRange(hsv, Scalar(0, 0, 50), Scalar(180, 255, 255), mask);
    
    // 形态学操作，去除噪声
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    
    return mask;
}

// 统计元素数目
int countElements(const Mat& mask) {
    // 查找轮廓
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // 过滤掉太小的轮廓（可能是噪声）
    int count = 0;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > 2) { // 阈值可以根据实际情况调整
            count++;
        }
    }
    
    return count;
}

// 保存两元素重叠图像
void saveOverlapImage(const Mat& mask1, const Mat& mask2, const string& outputPath, const Scalar& color) {
    Mat overlap = mask1 & mask2;
    Mat result = Mat::zeros(mask1.size(), CV_8UC3);
    result.setTo(color, overlap);
    imwrite(outputPath, result);
}

// 保存三元素重叠图像
void saveTripleOverlapImage(const Mat& mask1, const Mat& mask2, const Mat& mask3, const string& outputPath) {
    Mat tripleOverlap = mask1 & mask2 & mask3;
    Mat result = Mat::zeros(mask1.size(), CV_8UC3);
    result.setTo(Scalar(255, 255, 255), tripleOverlap); // 白色表示三者重叠
    imwrite(outputPath, result);
}
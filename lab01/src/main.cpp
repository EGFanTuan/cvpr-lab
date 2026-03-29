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
void saveDebugCrop(const Mat& original, const Mat& mask, const string& elem, const string& outputPath);

int main() {
    // 图像路径
    string inputPath = string(PROJECT_DIR) + "/input/";
    string outputPath = string(PROJECT_DIR) + "/output/";
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
        
        // 对比分割效果
        saveDebugCrop(image, mask, elem, outputPath);
        
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
    // 不需要了，保留完整矩形区域
    // Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    // morphologyEx(mask, mask, MORPH_OPEN, kernel);
    // morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    
    return mask;
}

// 统计元素数目
int countElements(const Mat& mask) {
    if (countNonZero(mask) == 0) return 0;

    // 前一版使用了“全局归一化(normalize)”，这是导致结果荒谬的元凶：
    // 当原图里有一个特别大的色块时，最大距离会被拉得极高
    // 按比例 0.3 截断时，所有面积较小的正常元素都被当做背景抹除了
    // 而重叠部分(Overlap)因为没有大色块，最高值很低，小颗粒反而存活了，这就导致重叠数 > 原元素数
    // 真的吗？

    // 尝试使用轻度“腐蚀(Erode)”或者绝对距离来断开微小的粘连。
    Mat processedMask;
    // 使用 3x3 的十字形形态学核，它的“切断”能力最轻柔，
    // 恰好能断开边缘只相连了 1 像素的相邻元素，又不会把小元素完全吃光。
    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
    erode(mask, processedMask, kernel);
    
    // 如果腐蚀导致图像全黑（比如全是 1x1 散点），就回退到原 mask
    if (countNonZero(processedMask) == 0) {
        processedMask = mask;
    }

    // 转换为 8 连通域计算独立斑块的数量
    Mat labels;
    int count = connectedComponents(processedMask, labels, 8) - 1; // 减去的 1 为黑色背景区域
    
    return count > 0 ? count : 0;
}

// 专门用来切除30x30块进行调试对比的函数
// 其实可能随机切更好？
void saveDebugCrop(const Mat& original, const Mat& mask, const string& elem, const string& outputPath) {
    Rect cropRect(0, 0, 0, 0);
    // 寻找一个带有前景元素的区域
    for (int y = 0; y < mask.rows - 30; ++y) {
        for (int x = 0; x < mask.cols - 30; ++x) {
            // 如果中心点有元素，就切这个 30x30 的块
            if (mask.at<uchar>(y + 15, x + 15) > 0) {
                cropRect = Rect(x, y, 30, 30);
                break;
            }
        }
        if (cropRect.width > 0) break;
    }
    
    if (cropRect.width == 0) return; // 如果全是黑的就算了

    Mat origCrop = original(cropRect);
    Mat maskCrop = mask(cropRect);
    
    // 模拟 countElements 里的腐蚀分离操作
    Mat processedMask;
    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
    erode(maskCrop, processedMask, kernel);
    if (countNonZero(processedMask) == 0) processedMask = maskCrop;

    // 为了肉眼方便观察，用最近邻插值（以防模糊）放大10倍，变成 300x300
    Mat visOrig, visMask, visProc;
    resize(origCrop, visOrig, Size(300, 300), 0, 0, INTER_NEAREST);
    resize(maskCrop, visMask, Size(300, 300), 0, 0, INTER_NEAREST);
    resize(processedMask, visProc, Size(300, 300), 0, 0, INTER_NEAREST);

    imwrite(outputPath + "debug_" + elem + "_1_orig.jpg", visOrig);
    imwrite(outputPath + "debug_" + elem + "_2_mask.jpg", visMask);
    imwrite(outputPath + "debug_" + elem + "_3_proc.jpg", visProc);
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

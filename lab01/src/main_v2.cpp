#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

bool isNonBlack(const Vec3b& color) {
    return color != Vec3b(0, 0, 0);
}

int main() {
    string inputPath = "../input/";
    string outputPath = "../output_v2/";
    system(("mkdir -p " + outputPath).c_str());
    
    Mat alImg = imread(inputPath + "Al.jpg");
    Mat feImg = imread(inputPath + "Fe.jpg");
    Mat pImg = imread(inputPath + "P.jpg");
    
    if (alImg.empty() || feImg.empty() || pImg.empty()) {
        cout << "无法读取输入图像" << endl;
        return -1;
    }
    
    cout << "=== 基于3x3块中心点检测 ===" << endl;
    
    Mat merged = alImg + feImg + pImg;
    
    int alCount = 0, feCount = 0, pCount = 0;
    int al_fe_Count = 0, al_p_Count = 0, fe_p_Count = 0;
    int al_fe_p_Count = 0;
    
    Mat alMask(alImg.rows, alImg.cols, CV_8UC1, Scalar(0));
    Mat feMask(alImg.rows, alImg.cols, CV_8UC1, Scalar(0));
    Mat pMask(alImg.rows, alImg.cols, CV_8UC1, Scalar(0));
    Mat vis(alImg.rows, alImg.cols, CV_8UC3, Scalar(0, 0, 0));
    
    for (int y = 0; y + 2 < merged.rows; y += 3) {
        for (int x = 0; x + 2 < merged.cols; x += 3) {
            Vec3b centerAl = alImg.at<Vec3b>(y + 1, x + 1);
            Vec3b centerFe = feImg.at<Vec3b>(y + 1, x + 1);
            Vec3b centerP = pImg.at<Vec3b>(y + 1, x + 1);
            
            bool hasAl = isNonBlack(centerAl);
            bool hasFe = isNonBlack(centerFe);
            bool hasP = isNonBlack(centerP);
            
            if (!hasAl && !hasFe && !hasP) continue;
            
            // 绘制 Mask
            if (hasAl) alMask(Rect(x, y, 3, 3)) = 255;
            if (hasFe) feMask(Rect(x, y, 3, 3)) = 255;
            if (hasP)  pMask(Rect(x, y, 3, 3)) = 255;
            
            // 统计数目 (包含重叠的累加)
            if (hasAl) alCount++;
            if (hasFe) feCount++;
            if (hasP) pCount++;
            if (hasAl && hasFe) al_fe_Count++;
            if (hasAl && hasP) al_p_Count++;
            if (hasFe && hasP) fe_p_Count++;
            if (hasAl && hasFe && hasP) al_fe_p_Count++;

            int type = (hasAl ? 1 : 0) * 1 + (hasFe ? 1 : 0) * 2 + (hasP ? 1 : 0) * 4;
            
            switch(type) {
                case 1: 
                    vis(Rect(x, y, 3, 3)) = Vec3b(255, 0, 0);
                    break;
                case 2: 
                    vis(Rect(x, y, 3, 3)) = Vec3b(0, 0, 255);
                    break;
                case 4: 
                    vis(Rect(x, y, 3, 3)) = Vec3b(0, 255, 0);
                    break;
                case 3: 
                    vis(Rect(x, y, 3, 3)) = Vec3b(255, 255, 0);
                    break;
                case 5: 
                    vis(Rect(x, y, 3, 3)) = Vec3b(255, 0, 255);
                    break;
                case 6: 
                    vis(Rect(x, y, 3, 3)) = Vec3b(0, 255, 255);
                    break;
                case 7: 
                    vis(Rect(x, y, 3, 3)) = Vec3b(255, 255, 255);
                    break;
            }
        }
    }
    
    cout << "\n=== 统计结果 ===" << endl;
    cout << "Al 元素数目: " << alCount << endl;
    cout << "Fe 元素数目: " << feCount << endl;
    cout << "P 元素数目: " << pCount << endl;
    cout << "Al 和 Fe 重叠数目: " << al_fe_Count << endl;
    cout << "Al 和 P 重叠数目: " << al_p_Count << endl;
    cout << "Fe 和 P 重叠数目: " << fe_p_Count << endl;
    cout << "Al、Fe 和 P 三者重叠数目: " << al_fe_p_Count << endl;
    
    imwrite(outputPath + "visualization.jpg", vis);
    imwrite(outputPath + "al_only.jpg", alMask);
    imwrite(outputPath + "fe_only.jpg", feMask);
    imwrite(outputPath + "p_only.jpg", pMask);
    
    Mat al_fe_overlap = alMask & feMask;
    Mat al_p_overlap = alMask & pMask;
    Mat fe_p_overlap = feMask & pMask;
    Mat al_fe_p_overlap = alMask & feMask & pMask;
    
    Mat al_fe_vis(alImg.rows, alImg.cols, CV_8UC3, Scalar(0, 0, 0));
    Mat al_p_vis(alImg.rows, alImg.cols, CV_8UC3, Scalar(0, 0, 0));
    Mat fe_p_vis(alImg.rows, alImg.cols, CV_8UC3, Scalar(0, 0, 0));
    Mat al_fe_p_vis(alImg.rows, alImg.cols, CV_8UC3, Scalar(0, 0, 0));
    
    al_fe_vis.setTo(Scalar(0, 255, 255), al_fe_overlap);
    al_p_vis.setTo(Scalar(255, 0, 255), al_p_overlap);
    fe_p_vis.setTo(Scalar(255, 255, 0), fe_p_overlap);
    al_fe_p_vis.setTo(Scalar(255, 255, 255), al_fe_p_overlap);
    
    imwrite(outputPath + "al_fe_overlap.jpg", al_fe_overlap);
    imwrite(outputPath + "al_p_overlap.jpg", al_p_overlap);
    imwrite(outputPath + "fe_p_overlap.jpg", fe_p_overlap);
    imwrite(outputPath + "al_fe_p_overlap.jpg", al_fe_p_overlap);
    
    imwrite(outputPath + "al_fe_overlap_vis.jpg", al_fe_vis);
    imwrite(outputPath + "al_p_overlap_vis.jpg", al_p_vis);
    imwrite(outputPath + "fe_p_overlap_vis.jpg", fe_p_vis);
    imwrite(outputPath + "al_fe_p_overlap_vis.jpg", al_fe_p_vis);
    
    cout << "\n结果已保存到 " << outputPath << endl;
    
    return 0;
}

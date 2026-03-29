#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    string imagePath = "input/picture.jpg";
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

    GaussianBlur(gray, gray, Size(5, 5), 1.5);

    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
                 gray.rows / 8,
                 100,
                 30,
                 80, 200);

    Mat result = image.clone();
    Mat edgeImage;
    Canny(gray, edgeImage, 50, 150);

    cout << "检测到 " << circles.size() << " 个钱币" << endl;
    cout << "========================================" << endl;

    for (size_t i = 0; i < circles.size(); i++) {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        int radius = c[2];

        circle(result, center, radius, Scalar(0, 255, 0), 2);
        circle(result, center, 3, Scalar(0, 0, 255), -1);

        cout << "钱币 " << (i + 1) << ":" << endl;
        cout << "  圆心坐标: (" << c[0] << ", " << c[1] << ")" << endl;
        cout << "  半径: " << radius << endl;
    }

    string outputDir = "output/";
    system("mkdir -p output");

    imwrite(outputDir + "result.jpg", result);
    imwrite(outputDir + "edge.jpg", edgeImage);

    imshow("原始图片", image);
    imshow("Canny边缘", edgeImage);
    imshow("检测结果", result);

    cout << "\n结果已保存到 " << outputDir << endl;
    cout << "按任意键退出..." << endl;
    waitKey(0);

    return 0;
}

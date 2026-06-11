#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

int main(int argc, char** argv){
  std::string input_path = std::string(PROJECT_DIR) + "/input/";
  auto images = std::vector<std::string>{
    input_path + "pic_01.jpg",
    input_path + "pic_02.jpg",
    input_path + "pic_03.jpg",
    input_path + "pic_04.jpg",
  };

  // read images
  std::vector<cv::Mat> mats;
  for (const auto& path : images) {
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
      std::cerr << "Failed to read image: " << path << std::endl;
      return -1;
    }
    mats.push_back(img);
  }

  cv::Ptr<cv::SIFT> detector = cv::SIFT::create();

  // create an instance of Stitcher
  cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);

  // merge images
  cv::Mat pano;
  cv::Stitcher::Status status = stitcher->stitch(mats, pano);
  if (status != cv::Stitcher::OK) {
    std::cerr << "Error during stitching, error code: " << int(status) << std::endl;
    return -1;
  }

  // save result
  std::string outputDir = std::string(PROJECT_DIR) + "/output/";
  std::error_code ec;
  std::filesystem::create_directories(outputDir, ec);
  if (ec) {
      std::cerr << "Failed to create output directory: " << outputDir << ", error: " << ec.message() << std::endl;
      return -1;
  }
  std::string output_path = std::string(PROJECT_DIR) + "/output/panorama.jpg";
  if (!cv::imwrite(output_path, pano)) {
    std::cerr << "Failed to save panorama image to: " << output_path << std::endl;
    return -1;
  }

  return 0;
}
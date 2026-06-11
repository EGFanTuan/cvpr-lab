#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

int main(int argc, char** argv){
  std::string input_path = std::string(PROJECT_DIR) + "/input/";

  // Define three groups of input images
  std::vector<std::vector<std::string>> imageGroups = {
    {
      input_path + "pic_01.jpg",
      input_path + "pic_02.jpg",
      input_path + "pic_03.jpg",
      input_path + "pic_04.jpg",
    },
    // {
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

  // Ensure output directory exists
  std::string outputDir = std::string(PROJECT_DIR) + "/output/";
  std::error_code ec;
  std::filesystem::create_directories(outputDir, ec);
  if (ec) {
      std::cerr << "Failed to create output directory: " << outputDir << ", error: " << ec.message() << std::endl;
      return -1;
  }

  // Process each group
  for (size_t g = 0; g < imageGroups.size(); ++g) {
    std::cout << "=== Processing group " << (g + 1) << " ===" << std::endl;

    // Read images for this group
    std::vector<cv::Mat> mats;
    for (const auto& path : imageGroups[g]) {
      cv::Mat img = cv::imread(path);
      if (img.empty()) {
        std::cerr << "Failed to read image: " << path << std::endl;
        return -1;
      }
      mats.push_back(img);
    }

    // Create SIFT detector and Stitcher
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);

    // Stitch images
    cv::Mat pano;
    cv::Stitcher::Status status = stitcher->stitch(mats, pano);
    if (status != cv::Stitcher::OK) {
      std::cerr << "Error during stitching group " << (g + 1)
                << ", error code: " << int(status) << std::endl;
    }

    // Save result
    std::string output_path = outputDir + "panorama_" + std::to_string(g + 1) + ".jpg";
    if (!cv::imwrite(output_path, pano)) {
      std::cerr << "Failed to save panorama image to: " << output_path << std::endl;
      return -1;
    }
    std::cout << "Saved: " << output_path << std::endl;
  }

  return 0;
}
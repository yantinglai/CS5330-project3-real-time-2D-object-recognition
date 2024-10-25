#ifndef FILTERS
#define FILTERS

#include <opencv2/opencv.hpp>

// Function to blur the image using a 5x5 filter
int blur5x5(cv::Mat &src, cv::Mat &dst);

// Task 1: To threshold the image into a binary image
int thresholding(cv::Mat &src, cv::Mat &dst, int threshold);

// Task 2: Perform closing (dilate followed by erode)
int closing(cv::Mat &src, cv::Mat &dst);

// Allow users to adjust the threshold dynamically using a trackbar
int adjustThreshold(cv::Mat &frame, int threshold);

// Helper function to save new object feature into CSV
int saveNewObject(cv::Mat &frame, cv::Mat &res, std::vector<float> &feature, char* dirName);

#endif

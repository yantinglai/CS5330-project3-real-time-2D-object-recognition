#include "filters.h"
#include <opencv2/opencv.hpp>

// Task 1: Threshold the image into a binary image
int thresholding(cv::Mat &src, cv::Mat &dst, int threshold) {
    cv::Mat gray, blurred;
    
    // Convert to grayscale
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur to reduce noise
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // Threshold the image
    cv::threshold(blurred, dst, threshold, 255, cv::THRESH_BINARY);

    return 0;
}

// Apply a 5x5 Gaussian blur as a separable filter
int blur5x5(cv::Mat &src, cv::Mat &dst) {
    cv::GaussianBlur(src, dst, cv::Size(5, 5), 0);
    return 0;
}

// Task 2: Perform closing (dilate followed by erode)
int closing(cv::Mat &src, cv::Mat &dst) {
    // Morphological closing operation (dilation followed by erosion)
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(src, dst, cv::MORPH_CLOSE, element);
    return 0;
}

// Adjust threshold dynamically using a trackbar
int adjustThreshold(cv::Mat &frame, int threshold) {
    cv::namedWindow("Adjust Threshold");
    int newThreshold = threshold;
    cv::createTrackbar("Threshold", "Adjust Threshold", &newThreshold, 255);
    while (true) {
        cv::Mat res;
        thresholding(frame, res, newThreshold);
        cv::imshow("Adjust Threshold", res);
        if (cv::waitKey(30) >= 0) {
            break;
        }
    }
    cv::destroyWindow("Adjust Threshold");
    return newThreshold;
}

#ifndef FETCHFEATURE_H
#define FETCHFEATURE_H

#include <opencv2/opencv.hpp>
#include <vector>

// Function declarations for fetchFeature.cpp

// Threshold the image, converting to grayscale and setting binary threshold
cv::Mat threshold(cv::Mat &image);

// Clean up the image using morphological operations (dilation, erosion)
cv::Mat cleanup(cv::Mat &image);

// Extract the largest regions from the binary image and return labeled regions
cv::Mat getRegions(cv::Mat &image, cv::Mat &labeledRegions, cv::Mat &stats, cv::Mat &centroids, std::vector<int> &topNLabels);

// Get the rotated bounding box for a region given centroid and orientation
cv::RotatedRect getBoundingBox(cv::Mat &region, double x, double y, double alpha);

// Draw a direction line on the image
void drawLine(cv::Mat &image, double x, double y, double alpha, cv::Scalar color);

// Draw an oriented bounding box around a region
void drawBoundingBox(cv::Mat &image, cv::RotatedRect boundingBox, cv::Scalar color);

// Calculate Hu Moments for feature extraction
void calcHuMoments(cv::Moments mo, std::vector<double> &huMoments);

#endif // FETCHFEATURE_H

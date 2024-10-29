/*
  Author: Yanting Lai
  Date: 2024-10-28
  CS 5330 Computer Vision
*/

#ifndef FETCHFEATURE_H
#define FETCHFEATURE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>

// Function declarations for fetchFeature.cpp

/**
* @brief Converts image to binary using threshold
* @param image Input image to be thresholded
* @return Binary image after thresholding
*/ 
cv::Mat threshold(cv::Mat &image);

/**
* @brief Alternative thresholding method with custom parameters
* @param image Input image
* @return Binary image after custom thresholding
*/
cv::Mat customeThreshold(cv::Mat &image);

/**
* @brief Cleans binary image using morphological operations
* @param image Binary input image
* @return Cleaned binary image after morphological operations
*/
cv::Mat cleanup(cv::Mat &image);

/**
* @brief Extracts and labels connected components from binary image
* @param image Binary input image
* @param labeledRegions Output matrix containing labeled regions
* @param stats Statistics for each labeled region
* @param centroids Centroids of labeled regions
* @param topNLabels Vector to store labels of largest regions
* @return Matrix containing labeled regions
*/
cv::Mat getRegions(cv::Mat &image, cv::Mat &labeledRegions, cv::Mat &stats, 
                  cv::Mat &centroids, std::vector<int> &topNLabels);

/**
* @brief Calculates rotated bounding box for a region
* @param region Binary region image
* @param x X-coordinate of centroid
* @param y Y-coordinate of centroid
* @param alpha Rotation angle
* @return RotatedRect object containing bounding box
*/
cv::RotatedRect getBoundingBox(cv::Mat &region, double x, double y, double alpha);

/**
* @brief Draws orientation line on image
* @param image Image to draw on
* @param x X-coordinate of line center
* @param y Y-coordinate of line center
* @param alpha Line angle
* @param color Line color
*/
void drawLine(cv::Mat &image, double x, double y, double alpha, cv::Scalar color);

/**
* @brief Draws rotated bounding box on image
* @param image Image to draw on
* @param boundingBox RotatedRect defining the box
* @param color Box color
*/
void drawBoundingBox(cv::Mat &image, cv::RotatedRect boundingBox, cv::Scalar color);

/**
* @brief Calculates Hu Moments for shape description
* @param mo Input moments
* @param huMoments Output vector of Hu Moments
*/
void calcHuMoments(cv::Moments mo, std::vector<double> &huMoments);

/**
* @brief Calculates cosine similarity between two feature vectors
* @param v1 First feature vector
* @param v2 Second feature vector
* @return Cosine similarity value
*/
double cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2);

/**
* @brief Structure for storing distance-label pairs in KNN
*/
struct DistanceLabel {
   double distance;
   std::string label;
   
   // Default constructor
   DistanceLabel() : distance(0.0), label("") {}
   
   // Constructor with parameters
   DistanceLabel(double d, std::string l) : distance(d), label(l) {}
   
   // Comparison operator for sorting
   bool operator<(const DistanceLabel& other) const {
       return distance < other.distance;
   }
};

/**
* @brief Calculates scaled Euclidean distance between feature vectors
* @param v1 First feature vector
* @param v2 Second feature vector
* @param stdevs Standard deviations for scaling
* @return Scaled Euclidean distance
*/
double scaled_euclidean_distance(const std::vector<float>& v1, 
                              const std::vector<float>& v2,
                              const std::vector<float>& stdevs);

/**
* @brief Calculates standard deviations of feature vectors
* @param feature_to_label Map of feature vectors to labels
* @return Vector of standard deviations
*/
std::vector<float> calculate_stdevs(const std::map<std::vector<float>, 
                                  std::string>& feature_to_label);

/**
* @brief Performs K-Nearest Neighbor classification
* @param query_vector Feature vector to classify
* @param feature_to_label Training data map
* @param stdevs Standard deviations for scaling
* @param k Number of neighbors to consider
* @param confidence_threshold Threshold for unknown classification
* @return Predicted label
*/
std::string improved_knn_classify(const std::vector<float>& query_vector,
                               const std::map<std::vector<float>, std::string>& feature_to_label,
                               const std::vector<float>& stdevs,
                               int k,
                               double confidence_threshold);

#endif // FETCHFEATURE_H
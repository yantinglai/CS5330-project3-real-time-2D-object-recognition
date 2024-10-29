#ifndef FETCHFEATURE_H
#define FETCHFEATURE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>

// Function declarations for fetchFeature.cpp

// Threshold the image, converting to grayscale and setting binary threshold
cv::Mat threshold(cv::Mat &image);

cv::Mat customeThreshold(cv::Mat &image);

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

double cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2);

// New additions for KNN implementation
struct DistanceLabel {
    double distance;
    std::string label;
    
    // Default constructor (needed for vector operations)
    DistanceLabel() : distance(0.0), label("") {}
    
    // Constructor with parameters
    DistanceLabel(double d, std::string l) : distance(d), label(l) {}
    
    // For sorting
    bool operator<(const DistanceLabel& other) const {
        return distance < other.distance;
    }
};

// Calculate scaled Euclidean distance
double scaled_euclidean_distance(const std::vector<float>& v1, const std::vector<float>& v2, const std::vector<float>& stdevs);

// Calculate standard deviations for feature scaling
std::vector<float> calculate_stdevs(const std::map<std::vector<float>, std::string>& feature_to_label);

// KNN method
std::string improved_knn_classify(const std::vector<float>& query_vector, 
                                const std::map<std::vector<float>, std::string>& feature_to_label,
                                const std::vector<float>& stdevs,
                                int k,
                                double confidence_threshold);

#endif // FETCHFEATURE_H
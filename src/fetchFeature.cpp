#include <stdlib.h>
#include <map>
#include <float.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "fetchFeature.h"
#include <numeric> 

using namespace cv;
using namespace std;

Mat threshold(Mat &image) {
    int THRESHOLD = 120; // Adjust threshold value to make it accurate
    Mat processedImage, grayscale;
    processedImage = Mat(image.size(), CV_8UC1);
    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    for (int i = 0; i < grayscale.rows; i++) {
        for (int j = 0; j < grayscale.cols; j++) {
            if (grayscale.at<uchar>(i, j) <= THRESHOLD) {
                processedImage.at<uchar>(i, j) = 255;
            } else {
                processedImage.at<uchar>(i, j) = 0;
            }
        }
    }
    return processedImage;
}

Mat customeThreshold(Mat &image) {
     Mat grayscale, processedImage;
    // Convert the image to grayscale
    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    // Optional: Apply histogram equalization to reduce shadow effects
    equalizeHist(grayscale, grayscale);

    // Apply adaptive thresholding to handle shadows
    adaptiveThreshold(grayscale, processedImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 15, 5);
    return processedImage;
}


Mat cleanup(Mat &image) {
    Mat processedImage;
    const Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15)); // Use smaller kernel
    morphologyEx(image, processedImage, MORPH_OPEN, kernel); // Apply opening to remove small noise
    return processedImage;
}

Mat getRegions(Mat &image, Mat &labeledRegions, Mat &stats, Mat &centroids, vector<int> &topNLabels) {
    Mat processedImage;
    int nLabels = connectedComponentsWithStats(image, labeledRegions, stats, centroids);
    Mat areas = Mat::zeros(1, nLabels - 1, CV_32S);
    Mat sortedIdx;

    for (int i = 1; i < nLabels; i++) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        areas.at<int>(i - 1) = area;
    }

    if (areas.cols > 0) {
        sortIdx(areas, sortedIdx, SORT_EVERY_ROW + SORT_DESCENDING);
    }

    vector<Vec3b> colors(nLabels, Vec3b(0, 0, 0));
    int N = 3; 
    N = (N < sortedIdx.cols) ? N : sortedIdx.cols;
    int THRESHOLD = 5000; 
    for (int i = 0; i < N; i++) {
        int label = sortedIdx.at<int>(i) + 1;
        if (stats.at<int>(label, CC_STAT_AREA) > THRESHOLD) {
            colors[label] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
            topNLabels.push_back(label);
        }
    }

    processedImage = Mat::zeros(labeledRegions.size(), CV_8UC3);
    for (int i = 0; i < processedImage.rows; i++) {
        for (int j = 0; j < processedImage.cols; j++) {
            int label = labeledRegions.at<int>(i, j);
            processedImage.at<Vec3b>(i, j) = colors[label];
        }
    }
    return processedImage;
}

RotatedRect getBoundingBox(Mat &region, double x, double y, double alpha) {
    int maxX = INT_MIN, minX = INT_MAX, maxY = INT_MIN, minY = INT_MAX;

    for (int i = 0; i < region.rows; i++) {
        for (int j = 0; j < region.cols; j++) {
            if (region.at<uchar>(i, j) == 255) {
                int projectedX = (i - x) * cos(alpha) + (j - y) * sin(alpha);
                int projectedY = -(i - x) * sin(alpha) + (j - y) * cos(alpha);
                maxX = max(maxX, projectedX);
                minX = min(minX, projectedX);
                maxY = max(maxY, projectedY);
                minY = min(minY, projectedY);
            }
        }
    }

    int lengthX = maxX - minX;
    int lengthY = maxY - minY;
    Point centroid = Point(x, y);
    Size size = Size(lengthX, lengthY);

    return RotatedRect(centroid, size, alpha * 180.0 / CV_PI + 90);
}

void drawLine(Mat &image, double x, double y, double alpha, Scalar color) {
    double length = 100.0;
    double dx = length * cos(alpha);  // X displacement based on the angle
    double dy = length * sin(alpha);  // Y displacement based on the angle

    double xPrime = x + dx;  // New x coordinate based on angle
    double yPrime = y + dy;  // New y coordinate based on angle

    arrowedLine(image, Point(x, y), Point(xPrime, yPrime), color, 15); // increase the arrowline thickness
}

void drawBoundingBox(Mat &image, RotatedRect boundingBox, Scalar color) {
    Point2f rect_points[4];
    boundingBox.points(rect_points);
    for (int i = 0; i < 4; i++) {
         line(image, rect_points[i], rect_points[(i + 1) % 4], color, 15); // increase the bouding box thickness
    }
}

void calcHuMoments(Moments mo, vector<double> &huMoments) {
    double hu[7]; 
    HuMoments(mo, hu);

    for (double d : hu) {
        huMoments.push_back(d);
    }
}

double cosine_similarity(const vector<float>& v1, const vector<float>& v2) {
    double dot_product = inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
    double norm1 = sqrt(inner_product(v1.begin(), v1.end(), v1.begin(), 0.0));
    double norm2 = sqrt(inner_product(v2.begin(), v2.end(), v2.begin(), 0.0));
    return dot_product / (norm1 * norm2);
}

std::vector<float> calculate_stdevs(const std::map<std::vector<float>, std::string>& feature_to_label) {
    // Get the dimension of feature vectors
    if (feature_to_label.empty()) {
        return std::vector<float>();
    }
    size_t dim = feature_to_label.begin()->first.size();
    
    // Calculate mean for each dimension
    std::vector<float> means(dim, 0.0);
    std::vector<float> squared_sums(dim, 0.0);
    int n = feature_to_label.size();
    
    for (const auto& pair : feature_to_label) {
        const auto& features = pair.first;
        for (size_t i = 0; i < dim; i++) {
            means[i] += features[i];
            squared_sums[i] += features[i] * features[i];
        }
    }
    
    // Calculate standard deviation
    std::vector<float> stdevs(dim);
    for (size_t i = 0; i < dim; i++) {
        means[i] /= n;
        float variance = (squared_sums[i] / n) - (means[i] * means[i]);
        stdevs[i] = std::sqrt(variance);
        // Avoid division by zero
        if (stdevs[i] < 1e-6) {
            stdevs[i] = 1.0;
        }
    }
    
    return stdevs;
}

// Calculate scaled Euclidean distance between two feature vectors
double scaled_euclidean_distance(const std::vector<float>& v1, const std::vector<float>& v2, const std::vector<float>& stdevs) {
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        double diff = (v1[i] - v2[i]) / stdevs[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::string improved_knn_classify(const std::vector<float>& query_vector, 
                                const std::map<std::vector<float>, std::string>& feature_to_label,
                                const std::vector<float>& stdevs,
                                int k,
                                double confidence_threshold) {
    vector<DistanceLabel> all_distances;
    
    // Calculate distances to all training examples
    for (const auto& pair : feature_to_label) {
        double distance = scaled_euclidean_distance(query_vector, pair.first, stdevs);
        all_distances.emplace_back(distance, pair.second);
    }
    
    // Sort by distance
    sort(all_distances.begin(), all_distances.end());
    
    // Take top K neighbors
    double sum_weights = 0;
    map<string, double> weighted_votes;
    
    // Use only k nearest neighbors
    int num_neighbors = min(k, (int)all_distances.size());
    
    // Calculate weights and votes
    for (int i = 0; i < num_neighbors; i++) {
        // Use inverse distance weighting
        double weight = 1.0 / (all_distances[i].distance + 1e-6);
        weighted_votes[all_distances[i].label] += weight;
        sum_weights += weight;
    }
    
    // Find the class with highest weighted vote
    string best_label;
    double max_vote = 0;
    
    // Print debug information
    cout << "\nKNN Debug Info (k=" << k << "):" << endl;
    for (const auto& vote : weighted_votes) {
        double normalized_vote = vote.second / sum_weights;
        cout << vote.first << ": " << (normalized_vote * 100) << "% confidence" << endl;
        
        if (normalized_vote > max_vote) {
            max_vote = normalized_vote;
            best_label = vote.first;
        }
    }
    
    // Check confidence threshold
    if (max_vote < confidence_threshold) {
        cout << "Low confidence classification (" << (max_vote * 100) << "% < " 
             << (confidence_threshold * 100) << "%)" << endl;
        return "Unknown";
    }
    
    cout << "Selected class: " << best_label << " with " 
         << (max_vote * 100) << "% confidence" << endl;
    
    return best_label;
}
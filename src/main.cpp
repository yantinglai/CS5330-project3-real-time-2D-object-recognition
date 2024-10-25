#include <opencv2/opencv.hpp>
#include "fetchFeature.h"
#include <iostream>
#include "csv_util.h"

using namespace cv;
using namespace std;

int main() {
    // Path to your images
    std::string img1Path = "/Users/sundri/Desktop/CS5330/Project3/IMG_5531.JPG";
    std::string img2Path = "/Users/sundri/Desktop/CS5330/Project3/IMG_5532.JPG";

    // Load the images
    Mat img1 = imread(img1Path);
    Mat img2 = imread(img2Path);

    if (img1.empty() || img2.empty()) {
        cerr << "Error: Could not load one or more images." << endl;
        return -1;
    }

    // Threshold and cleanup the images
    Mat thresh1 = threshold(img1);
    Mat thresh2 = threshold(img2);
    Mat clean1 = cleanup(thresh1);
    Mat clean2 = cleanup(thresh2);

    // Get the largest 3 regions from both images
    Mat labeledRegions1, stats1, centroids1;
    Mat labeledRegions2, stats2, centroids2;
    vector<int> topNLabels1, topNLabels2;

    Mat regionMap1 = getRegions(clean1, labeledRegions1, stats1, centroids1, topNLabels1);
    Mat regionMap2 = getRegions(clean2, labeledRegions2, stats2, centroids2, topNLabels2);

    // Process each region in Image 1
for (int n = 0; n < topNLabels1.size(); n++) {
    int label = topNLabels1[n];
    Mat region = (labeledRegions1 == label);

    // Compute moments, center, and orientation
    Moments mo = moments(region, true);
    double centroidX = centroids1.at<double>(label, 0);
    double centroidY = centroids1.at<double>(label, 1);
    double alpha = 0.5 * atan2(2 * mo.mu11, mo.mu20 - mo.mu02);

    // Get the oriented bounding box for the region
    RotatedRect boundingBox = getBoundingBox(region, centroidX, centroidY, alpha);

    // Draw the bounding box and direction line
    drawLine(img1, centroidX, centroidY, alpha, Scalar(0, 0, 255));
    drawBoundingBox(img1, boundingBox, Scalar(0, 255, 0));

    // Calculate and print Hu Moments as the feature vector
    vector<double> huMoments;
    calcHuMoments(mo, huMoments);
    cout << "Feature vector (Hu Moments) for region " << n + 1 << " in Image 1: ";
    for (const auto& val : huMoments) {
        cout << val << " ";
    }
    cout << endl; // Move to the next line for the next region
}

// Process each region in Image 2
for (int n = 0; n < topNLabels2.size(); n++) {
    int label = topNLabels2[n];
    Mat region = (labeledRegions2 == label);

    // Compute moments, center, and orientation
    Moments mo = moments(region, true);
    double centroidX = centroids2.at<double>(label, 0);
    double centroidY = centroids2.at<double>(label, 1);
    double alpha = 0.5 * atan2(2 * mo.mu11, mo.mu20 - mo.mu02);

    // Get the oriented bounding box for the region
    RotatedRect boundingBox = getBoundingBox(region, centroidX, centroidY, alpha);

    // Draw the bounding box and direction line
    drawLine(img2, centroidX, centroidY, alpha, Scalar(0, 0, 255));
    drawBoundingBox(img2, boundingBox, Scalar(0, 255, 0));

    // Calculate and print Hu Moments as the feature vector
    vector<double> huMoments;
    calcHuMoments(mo, huMoments);
    cout << "Feature vector (Hu Moments) for region " << n + 1 << " in Image 2: ";
    for (const auto& val : huMoments) {
        cout << val << " ";
    }
    cout << endl; // Move to the next line for the next region
    }

    // Display the original, thresholded, cleaned, and segmented images
    imshow("Original Image 1", img1);
    imshow("Thresholded Image 1", thresh1);
    imshow("Cleaned Image 1", clean1);
    imshow("Segmented Regions Image 1", regionMap1);

    imshow("Original Image 2", img2);
    imshow("Thresholded Image 2", thresh2);
    imshow("Cleaned Image 2", clean2);
    imshow("Segmented Regions Image 2", regionMap2);

    // Wait for a key press before exiting
    waitKey(0);
    return 0;
}

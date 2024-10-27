#include <opencv2/opencv.hpp>
#include "fetchFeature.h"
#include <iostream>
#include "csv_util.h"

using namespace cv;
using namespace std;

int main() {
    // Paths to your images (updated to handle 4 images)
    std::string imgPaths[] = {
        "/Users/sundri/Desktop/CS5330/Project3/IMG_5572.JPG",
        "/Users/sundri/Desktop/CS5330/Project3/IMG_5573.JPG",
        "/Users/sundri/Desktop/CS5330/Project3/IMG_5574.JPG",
        "/Users/sundri/Desktop/CS5330/Project3/IMG_5575.JPG"
    };

    // Vector to hold the images
    vector<Mat> images(4);
    vector<Mat> threshImages(4);
    vector<Mat> cleanImages(4);
    vector<Mat> labeledRegions(4), stats(4), centroids(4);
    vector<vector<int>> topNLabels(4);

    std::string csvFilename = "/Users/sundri/Desktop/CS5330/Project3/feature_vectors.csv";
    std::vector<float> empty_vector;
    append_image_data_csv(csvFilename, "empty", empty_vector, 1); // use csv util to clean data

    // Load and process all 4 images
    for (int i = 0; i < 4; ++i) {
        images[i] = imread(imgPaths[i]);
        if (images[i].empty()) {
            cerr << "Error: Could not load image " << imgPaths[i] << endl;
            return -1;
        }

        // Threshold and cleanup the images
        threshImages[i] = threshold(images[i]);
        cleanImages[i] = cleanup(threshImages[i]);

        // Get the largest 3 regions from each image
        Mat regionMap = getRegions(cleanImages[i], labeledRegions[i], stats[i], centroids[i], topNLabels[i]);

        // Process each region in the current image
        for (int n = 0; n < topNLabels[i].size(); n++) {
            int label = topNLabels[i][n];
            Mat region = (labeledRegions[i] == label);

            // Compute moments, center, and orientation
            Moments mo = moments(region, true);
            double centroidX = centroids[i].at<double>(label, 0);
            double centroidY = centroids[i].at<double>(label, 1);
            double alpha = 0.5 * atan2(2 * mo.mu11, mo.mu20 - mo.mu02);

            // Get the oriented bounding box for the region
            RotatedRect boundingBox = getBoundingBox(region, centroidX, centroidY, alpha);

            // Draw the bounding box and direction line
            drawLine(images[i], centroidX, centroidY, alpha, Scalar(0, 0, 255)); // Arrow center
            drawBoundingBox(images[i], boundingBox, Scalar(0, 255, 0));

            // Calculate the percent filled and height/width ratio
            double area = boundingBox.size.width * boundingBox.size.height;
            double percentFilled = mo.m00 / area;
            double heightWidthRatio = boundingBox.size.height / boundingBox.size.width;

            // Calculate and print Hu Moments as the feature vector
            vector<double> huMoments;
            calcHuMoments(mo, huMoments);
            
            // Create feature vector (Hu Moments + Percentage filled + Height/width)
            vector<double> featureVector = huMoments;
            featureVector.push_back(percentFilled);
            featureVector.push_back(heightWidthRatio);
            
            // Write to CSV by calling append_image_data_csv
            // Convert featureVector from double to float for CSV
            vector<float> featureVectorFloat(featureVector.begin(), featureVector.end());

            // Write to CSV by calling append_image_data_csv
            append_image_data_csv(csvFilename, imgPaths[i], featureVectorFloat, 0); 
        }

        // Display the images (original, thresholded, cleaned, and segmented)
        imshow("Original Image " + to_string(i + 1), images[i]);
        imshow("Thresholded Image " + to_string(i + 1), threshImages[i]);
        imshow("Cleaned Image " + to_string(i + 1), cleanImages[i]);
        imshow("Segmented Regions Image " + to_string(i + 1), regionMap);
    }

    // Wait for a key press before exiting
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include "fetchFeature.h"
#include "csv_util.h"

using namespace cv;
using namespace std;
namespace fs = std::__fs::filesystem;

// Global variables for mouse callback
Point clickedPoint;
bool userClicked = false;

// Mouse callback function to capture mouse clicks
void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        clickedPoint = Point(x, y);
        userClicked = true;
    }
}

int main() {
    // Path to the base directory containing object folders
    std::string baseDir = "/Users/sundri/Desktop/CS5330/Project3/db/";
    std::string csvFilename = "/Users/sundri/Desktop/CS5330/Project3/feature_vectors.csv";
    std::vector<float> empty_vector;
    append_image_data_csv(csvFilename, "empty", empty_vector, 1); // Clean CSV file

    namedWindow("Processed Image");
    setMouseCallback("Processed Image", onMouse);

    // Loop through all object directories
    for (const auto& objectDir : fs::directory_iterator(baseDir)) {
        if (!fs::is_directory(objectDir)) continue;

        std::string trainingDir = objectDir.path().string() + "/training";

        // Loop through all images in the training folder
        for (const auto& entry : fs::directory_iterator(trainingDir)) {
            std::string imgPath = entry.path().string();
            Mat originalImage = imread(imgPath);
            if (originalImage.empty()) {
                cerr << "Error: Could not load image " << imgPath << endl;
                continue;
            }

            // Process image
            Mat threshImage = threshold(originalImage);
            Mat cleanImage = cleanup(threshImage);
            
            // Get regions
            Mat labeledRegions, stats, centroids;
            vector<int> topNLabels;
            Mat regionMap = getRegions(cleanImage, labeledRegions, stats, centroids, topNLabels);

            // **Find the largest region**
            int largestRegionLabel = -1;
            double maxArea = 0;
            for (int n = 0; n < topNLabels.size(); n++) {
                int label = topNLabels[n];
                double area = stats.at<int>(label, CC_STAT_AREA);
                if (area > maxArea) {
                    maxArea = area;
                    largestRegionLabel = label;
                }
            }

            // If no valid region found, continue to next image
            if (largestRegionLabel == -1) {
                cerr << "No valid region found for image: " << imgPath << endl;
                continue;
            }

            // Process the largest region
            Mat displayImage = originalImage.clone();
            Mat region = (labeledRegions == largestRegionLabel);

            // Compute moments and features for the largest region
            Moments mo = moments(region, true);
            double centroidX = centroids.at<double>(largestRegionLabel, 0);
            double centroidY = centroids.at<double>(largestRegionLabel, 1);
            double alpha = 0.5 * atan2(2 * mo.mu11, mo.mu20 - mo.mu02);

            // Get and draw bounding box
            RotatedRect boundingBox = getBoundingBox(region, centroidX, centroidY, alpha);
            drawLine(displayImage, centroidX, centroidY, alpha, Scalar(0, 0, 255));
            drawBoundingBox(displayImage, boundingBox, Scalar(0, 255, 0));

            // Calculate features
            double area = boundingBox.size.width * boundingBox.size.height;
            double percentFilled = mo.m00 / area;
            double heightWidthRatio = boundingBox.size.height / boundingBox.size.width;

            vector<double> huMoments;
            calcHuMoments(mo, huMoments);

            vector<double> featureVector = huMoments;
            featureVector.push_back(percentFilled);
            featureVector.push_back(heightWidthRatio);

            vector<float> featureVectorFloat(featureVector.begin(), featureVector.end());

            // Show image and wait for click
            imshow("Processed Image", displayImage);
            userClicked = false;
            
            // Wait for mouse click or 'q' to quit
            while (!userClicked) {
                char key = (char)waitKey(10);
                if (key == 'q' || key == 'Q') {
                    return 0;  // Exit program
                }
            }

            // Get label from user
            std::string objectLabel;
            cout << "Enter a label for the selected object: ";  // Ignore leftover newline from previous input
            getline(cin, objectLabel);  // Use getline to capture the full label, including spaces

            // Add label to image and CSV
            append_image_data_csv(csvFilename, objectLabel, featureVectorFloat, 0);
            
            // Enhanced text parameters for better visibility
            double fontScale = 2.5;  // Increased font size
            int thickness = 6;       // Increased thickness for bold text
            int fontFace = FONT_HERSHEY_DUPLEX;  // Changed font for better boldness
            
            // Get text size to center it above the object
            int baseline = 0;
            Size textSize = getTextSize(objectLabel, fontFace, fontScale, thickness, &baseline);
            
            // Calculate position to center text above object
            Point textOrg(
                centroidX - (textSize.width / 2),    // Center horizontally
                centroidY - 20 - textSize.height     // Place above object with padding
            );
            
            // Draw bold white text with black outline for better visibility
            putText(displayImage, objectLabel, textOrg, 
                fontFace, fontScale, Scalar(0, 0, 0), thickness + 2); // Black outline
            putText(displayImage, objectLabel, textOrg, 
                fontFace, fontScale, Scalar(255, 255, 255), thickness); // White fill

            // Show labeled image briefly
            imshow("Processed Image", displayImage);
            waitKey(500);  // Show labeled image for 500ms before moving to next region
        }
    }

    return 0;
}

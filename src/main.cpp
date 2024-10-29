#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <map>
#include <iomanip>  // for setw in printing
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
    std::map<std::vector<float>, std::string> feature_to_label = read_feature_vectors_and_labels(csvFilename);

    // Define mapping from directory names to class names
    std::map<std::string, std::string> dir_to_class = {
        {"object1", "nail clipper"},
        {"object2", "remote control"},
        {"object3", "candle lid"},
        {"object4", "shaver"},
        {"object5", "mouse"}
    };

    // Initialize confusion matrix with actual class names
    std::vector<std::string> classes = {"nail clipper", "remote control", "candle lid", "shaver", "mouse"};
    const int num_classes = classes.size();
    std::vector<std::vector<int>> confusion_matrix(num_classes, std::vector<int>(num_classes, 0));
    
    // Create map from class names to indices
    std::map<std::string, int> class_to_index;
    for (int i = 0; i < classes.size(); i++) {
        class_to_index[classes[i]] = i;
    }

    namedWindow("Processed Image");
    setMouseCallback("Processed Image", onMouse);

    // Debug: Print classes we're looking for
    cout << "Classes we're tracking:\n";
    for (const auto& cls : classes) {
        cout << "'" << cls << "'" << endl;
    }

    // Loop through all object directories
    for (const auto& objectDir : fs::directory_iterator(baseDir)) {
        if (!fs::is_directory(objectDir)) continue;

        std::string matchingDir = objectDir.path().string() + "/matching";
        std::string dir_name = objectDir.path().filename().string(); // Gets "object1", "object2", etc.
        std::string true_label = dir_to_class[dir_name]; // Convert to actual class name

        cout << "\nProcessing directory: '" << dir_name << "' (Class: " << true_label << ")" << endl;

        // Loop through all images in the matching folder
        for (const auto& entry : fs::directory_iterator(matchingDir)) {
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

            // Find the largest region
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

            // Find the most similar feature vector in the training set
            double bestSimilarity = 0.0;
            string bestLabel;
            for (const auto& pair : feature_to_label) {
                double similarity = cosine_similarity(featureVectorFloat, pair.first);
                if (similarity > bestSimilarity) {
                    bestSimilarity = similarity;
                    bestLabel = pair.second;
                }
            }

            // Debug: Print classification result
            cout << "True label: '" << true_label << "'" << endl;
            cout << "Predicted label: '" << bestLabel << "'" << endl;
            cout << "Similarity: " << bestSimilarity << endl;

            // Update confusion matrix
            if (class_to_index.find(true_label) != class_to_index.end() && 
                class_to_index.find(bestLabel) != class_to_index.end()) {
                int true_idx = class_to_index[true_label];
                int pred_idx = class_to_index[bestLabel];
                confusion_matrix[true_idx][pred_idx]++;
                cout << "Updated confusion matrix at [" << true_idx << "][" << pred_idx << "]" << endl;
            }

            // Show image and display the label
            imshow("Processed Image", displayImage);
            userClicked = false;
            
            // Wait for mouse click or 'q' to quit
            while (!userClicked) {
                char key = (char)waitKey(10);
                if (key == 'q' || key == 'Q') {
                    // Print final confusion matrix before exiting
                    cout << "\nFinal Confusion Matrix:\n";
                    cout << "Predicted →\n";
                    cout << "Actual ↓\n\t";
                    for (const auto& cls : classes) {
                        cout << setw(15) << cls << " ";
                    }
                    cout << "\n";
                    for (int i = 0; i < num_classes; i++) {
                        cout << setw(15) << classes[i] << " ";
                        for (int j = 0; j < num_classes; j++) {
                            cout << setw(15) << confusion_matrix[i][j] << " ";
                        }
                        cout << "\n";
                    }
                    return 0;
                }
            }

            // Enhanced text parameters for better visibility
            double fontScale = 2.5;
            int thickness = 6;
            int fontFace = FONT_HERSHEY_DUPLEX;
            
            // Get text size to center it above the object
            int baseline = 0;
            Size textSize = getTextSize(bestLabel, fontFace, fontScale, thickness, &baseline);
            
            // Calculate position to center text above object
            Point textOrg(
                centroidX - (textSize.width / 2),
                centroidY - 20 - textSize.height
            );
            
            // Draw bold white text with black outline for better visibility
            putText(displayImage, bestLabel, textOrg, 
                fontFace, fontScale, Scalar(0, 0, 0), thickness + 2);
            putText(displayImage, bestLabel, textOrg, 
                fontFace, fontScale, Scalar(255, 255, 255), thickness);

            // Show labeled image
            imshow("Processed Image", displayImage);
            waitKey(2000);
        }
    }

    // Print final confusion matrix
    cout << "\nFinal Confusion Matrix:\n";
    cout << "Predicted →\n";
    cout << "Actual ↓\n\t";
    for (const auto& cls : classes) {
        cout << setw(15) << cls << " ";
    }
    cout << "\n";
    for (int i = 0; i < num_classes; i++) {
        cout << setw(15) << classes[i] << " ";
        for (int j = 0; j < num_classes; j++) {
            cout << setw(15) << confusion_matrix[i][j] << " ";
        }
        cout << "\n";
    }

    // Calculate and print accuracy
    int correct = 0;
    int total = 0;
    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < num_classes; j++) {
            if (i == j) correct += confusion_matrix[i][j];
            total += confusion_matrix[i][j];
        }
    }
    
    double accuracy = total > 0 ? (double)correct / total : 0.0;
    cout << "\nOverall Accuracy: " << (accuracy * 100) << "%\n";

    return 0;
}
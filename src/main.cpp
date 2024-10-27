#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include "fetchFeature.h"
#include "csv_util.h"

using namespace cv;
using namespace std;
namespace fs = std::__fs::filesystem;


int main() {
    // Path to the base directory containing object folders
    std::string baseDir = "/Users/sundri/Desktop/CS5330/Project3/db/";

    // CSV file for storing feature vectors with labels
    std::string csvFilename = "/Users/sundri/Desktop/CS5330/Project3/feature_vectors.csv";
    std::vector<float> empty_vector;
    append_image_data_csv(csvFilename, "empty", empty_vector, 1); // Clean CSV file

    // Loop through all object directories (object1 to object5)
    for (const auto& objectDir : fs::directory_iterator(baseDir)) {
        if (!fs::is_directory(objectDir)) continue; // Skip non-directory entries
        
        std::string trainingDir = objectDir.path().string() + "/training"; // Path to training folder
        
        // Loop through all images in the training folder
        for (const auto& entry : fs::directory_iterator(trainingDir)) {
            std::string imgPath = entry.path().string();  // Get the image path

            // Load the image
            Mat image = imread(imgPath);
            if (image.empty()) {
                cerr << "Error: Could not load image " << imgPath << endl;
                continue;
            }

            // Threshold and cleanup the image
            Mat threshImage = threshold(image);
            imshow("Thresholded Image", threshImage); // Display thresholded image
            Mat cleanImage = cleanup(threshImage);
             imshow("cleanImage Image", cleanImage); // Display cleaned up image

            // Get the largest regions
            Mat labeledRegions, stats, centroids;
            vector<int> topNLabels;
            Mat regionMap = getRegions(cleanImage, labeledRegions, stats, centroids, topNLabels);
            imshow("regionMap", regionMap); // Display segmented image

            // Process each region (your existing logic)
            for (int n = 0; n < topNLabels.size(); n++) {
                int label = topNLabels[n];
                Mat region = (labeledRegions == label);

                // Compute moments, center, and orientation
                Moments mo = moments(region, true);
                double centroidX = centroids.at<double>(label, 0);
                double centroidY = centroids.at<double>(label, 1);
                double alpha = 0.5 * atan2(2 * mo.mu11, mo.mu20 - mo.mu02);

                // Get the oriented bounding box for the region
                RotatedRect boundingBox = getBoundingBox(region, centroidX, centroidY, alpha);

                // Draw the bounding box and direction line (optional for debugging)
                drawLine(image, centroidX, centroidY, alpha, Scalar(0, 0, 255));
                drawBoundingBox(image, boundingBox, Scalar(0, 255, 0));

                // Calculate the percent filled and height/width ratio
                double area = boundingBox.size.width * boundingBox.size.height;
                double percentFilled = mo.m00 / area;
                double heightWidthRatio = boundingBox.size.height / boundingBox.size.width;

                // Calculate Hu Moments as the feature vector
                vector<double> huMoments;
                calcHuMoments(mo, huMoments);

                // Create feature vector (Hu Moments + Percentage filled + Height/width)
                vector<double> featureVector = huMoments;
                featureVector.push_back(percentFilled);
                featureVector.push_back(heightWidthRatio);

                // Convert feature vector to float for CSV
                vector<float> featureVectorFloat(featureVector.begin(), featureVector.end());

                // **Modification**: Wait for 'N' key press to label the object
                if (waitKey(30) == 'N') {
                    std::string label;
                    cout << "Enter a label for this object: ";
                    cin >> label;
                    append_image_data_csv(csvFilename, label, featureVectorFloat, 0);
                }
            }

            // Optionally display the processed image (for debugging)
            imshow("Processed Image", image);

            // Exit on 'q' key press
            if (waitKey(30) == 'q') {
                break;
            }
        }
    }

    return 0;
}

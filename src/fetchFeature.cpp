#include <stdlib.h>
#include <map>
#include <float.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "fetchFeature.h"

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

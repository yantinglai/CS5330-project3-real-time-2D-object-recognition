/*
  Author: Yanting Lai
  Date: 2024-10-28
  CS 5330 Computer Vision
*/

#include "csv_util.h"
#include <fstream>
#include <iostream>
#include <sstream>

// Append a line of image data to a CSV file
int append_image_data_csv(const std::string &filename, const std::string &image_filename, const std::vector<float> &image_data, int reset_file) {
    std::ofstream file;

    // Open the file in truncate mode if reset_file is set to 1, otherwise open in append mode
    if (reset_file) {
        file.open(filename, std::ofstream::out | std::ofstream::trunc);
    } else {
        file.open(filename, std::ofstream::out | std::ofstream::app);
    }

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return -1;
    }

    // Write the image filename to the file
    file << image_filename;

    // Write the image data (features) to the file, separated by commas
    for (const auto &val : image_data) {
        file << "," << val;
    }

    // Add a newline at the end
    file << "\n";

    // Close the file
    file.close();

    return 0; // Return 0 for success
}

// Read image data from a CSV file
int read_image_data_csv(const std::string &filename, std::vector<std::string> &filenames, std::vector<std::vector<float>> &data, int echo_file) {
    std::ifstream file(filename);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return -1;
    }

    std::string line;

    // Read each line of the file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string file_name;
        std::getline(ss, file_name, ',');

        // Store the filename
        filenames.push_back(file_name);

        // Store the feature data
        std::vector<float> features;
        std::string feature;
        while (std::getline(ss, feature, ',')) {
            features.push_back(std::stof(feature)); // Convert string to float
        }

        data.push_back(features); // Add the features to the data vector
    }

    // Optionally echo the file content for debugging
    if (echo_file) {
        for (const auto &row : data) {
            for (const auto &val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0; // Return 0 for success
}

std::map<std::vector<float>, std::string> read_feature_vectors_and_labels(const std::string& csv_filename) {
    std::map<std::vector<float>, std::string> feature_to_label;
    
    // Open the CSV file
    std::ifstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + csv_filename);
    }

    // Read each line of data
    std::string line;
    // Skip the first line (empty line)
    std::getline(csv_file, line);

    while (std::getline(csv_file, line)) {
        std::stringstream line_stream(line);
        std::string item;
        std::vector<float> feature_vector;

        // Read the label
        std::getline(line_stream, item, ',');
        std::string label = item;

        // Read the feature vector
        while (std::getline(line_stream, item, ',')) {
            if (!item.empty()) {
                feature_vector.push_back(std::stof(item));
            }
        }

        // Only add to the map if the feature vector is not empty
        if (!feature_vector.empty()) {
            feature_to_label[feature_vector] = label;
        }
    }

    csv_file.close();
    return feature_to_label;
}

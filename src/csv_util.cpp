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

// Load feature vectors from CSV file
void loadFeatureVectors(const string& csvFilename, vector<vector<float>>& featureVectors, vector<string>& labels) {
    ifstream file(csvFilename);
    string line, label;

    while (getline(file, line)) {
        stringstream ss(line);
        vector<float> featureVector;
        string value;

        getline(ss, label, ',');  // First element is the label
        labels.push_back(label);  // Store label

        while (getline(ss, value, ',')) {
            featureVector.push_back(stof(value));  // Convert each value to float
        }

        featureVectors.push_back(featureVector);  // Store feature vector
    }
}
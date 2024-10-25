#include "csv_util.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>

// Append a line of image data to a CSV file
int append_image_data_csv(char *filename, char *image_filename, std::vector<float> &image_data, int reset_file) {
    std::ofstream file;
    if (reset_file) {
        file.open(filename, std::ofstream::out | std::ofstream::trunc);
    } else {
        file.open(filename, std::ofstream::out | std::ofstream::app);
    }

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return -1;
    }

    file << image_filename;
    for (const auto &val : image_data) {
        file << "," << val;
    }
    file << "\n";
    file.close();

    return 0;
}

// Read image data from a CSV file
int read_image_data_csv(char *filename, std::vector<char *> &filenames, std::vector<std::vector<float>> &data, int echo_file) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return -1;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string file_name;
        std::getline(ss, file_name, ',');

        char *fname = new char[file_name.size() + 1];
        std::strcpy(fname, file_name.c_str());
        filenames.push_back(fname);

        std::vector<float> features;
        std::string feature;
        while (std::getline(ss, feature, ',')) {
            features.push_back(std::stof(feature));
        }
        data.push_back(features);
    }

    if (echo_file) {
        for (const auto &row : data) {
            for (const auto &val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}

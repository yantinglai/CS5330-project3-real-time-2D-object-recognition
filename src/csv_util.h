#ifndef CSV_UTIL_H
#define CSV_UTIL_H

#include <string>
#include <vector>
#include <map>

// Function to append a line of image data to a CSV file
int append_image_data_csv(const std::string &filename, const std::string &image_filename, const std::vector<float> &image_data, int reset_file);

// Function to read image data from a CSV file
int read_image_data_csv(const std::string &filename, std::vector<std::string> &filenames, std::vector<std::vector<float>> &data, int echo_file);

std::map<std::vector<float>, std::string> read_feature_vectors_and_labels(const std::string& csv_filename);

#endif // CSV_UTIL_H

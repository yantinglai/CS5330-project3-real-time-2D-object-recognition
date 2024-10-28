#ifndef CSV_UTIL_H
#define CSV_UTIL_H

#include <string>
#include <vector>

// Function to append a line of image data to a CSV file
int append_image_data_csv(const std::string &filename, const std::string &image_filename, const std::vector<float> &image_data, int reset_file);

// Function to read image data from a CSV file
int read_image_data_csv(const std::string &filename, std::vector<std::string> &filenames, std::vector<std::vector<float>> &data, int echo_file);

void loadFeatureVectors(const std::string& csvFilename, std::vector<std::vector<float>>& featureVectors, std::vector<std::string>& labels);

#endif // CSV_UTIL_H

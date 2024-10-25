#ifndef CSV_UTIL_H
#define CSV_UTIL_H

#include <vector>
#include <string>

// Append a line of image data to a CSV file
int append_image_data_csv(char *filename, char *image_filename, std::vector<float> &image_data, int reset_file = 0);

// Read image data from a CSV file
int read_image_data_csv(char *filename, std::vector<char *> &filenames, std::vector<std::vector<float>> &data, int echo_file = 0);

#endif

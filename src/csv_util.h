/* Author: Yanting Lai
   Date: 2024-10-28
   CS 5330 Computer Vision 
*/

#ifndef CSV_UTIL_H
#define CSV_UTIL_H

#include <string>
#include <vector>
#include <map>

/**
 * Appends image feature data to a CSV file.
 * 
 * @param filename The path to the CSV file to write to
 * @param image_filename The name/path of the image being processed
 * @param image_data Vector containing extracted feature values from the image
 * @param reset_file If 1, clear existing file content before writing; if 0, append
 * @return 0 on success, non-zero on failure
 */
int append_image_data_csv(const std::string &filename, const std::string &image_filename, const std::vector<float> &image_data, int reset_file);

/**
 * Reads image data from a CSV file into vectors.
 * 
 * @param filename The path to the CSV file to read from
 * @param filenames Output vector to store image filenames/paths
 * @param data Output vector of vectors to store feature data for each image
 * @param echo_file If 1, print data while reading; if 0, read silently
 * @return 0 on success, non-zero on failure
 */
int read_image_data_csv(const std::string &filename, std::vector<std::string> &filenames, std::vector<std::vector<float>> &data, int echo_file);

/**
 * Reads feature vectors and their corresponding labels from a CSV file.
 * 
 * @param csv_filename The path to the CSV file containing feature vectors and labels
 * @return A map with feature vectors as keys and their labels as values
 */
std::map<std::vector<float>, std::string> read_feature_vectors_and_labels(const std::string& csv_filename);

#endif // CSV_UTIL_H
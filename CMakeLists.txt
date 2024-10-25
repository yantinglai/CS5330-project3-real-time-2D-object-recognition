cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Project3)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add executable target for main.cpp
add_executable(Project3 src/main.cpp src/filters.cpp src/fetchFeature.cpp src/csv_util.cpp)

# Link OpenCV libraries to your executable
target_link_libraries(Project3 ${OpenCV_LIBS})

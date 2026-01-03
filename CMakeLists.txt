cmake_minimum_required(VERSION 3.10)
project(MacTensor)

# standard stuff, c++17 cause i need newer features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# trying to find the apple accelerate framework
# if this fails, im screwed cause i need the M1 speed
find_library(ACCELERATE_FRAMEWORK Accelerate)

if(NOT ACCELERATE_FRAMEWORK)
    message(FATAL_ERROR "yo accelerate not found?? are u even on a mac?")
endif()

# tell cmake where my headers are living
include_directories(include)

# source files to compile
set(SOURCES
    src/main.cpp
    src/core/matrix.cpp
    src/ml/linear_regression.cpp
    src/ml/logistic_regression.cpp 
    src/ml/decision_tree.cpp
    src/ml/gradient_boosting.cpp
    src/ml/kmeans.cpp
    src/ml/pca.cpp 
)

# make the executable
add_executable(MacTensor ${SOURCES})

# link the magic apple math library
target_link_libraries(MacTensor ${ACCELERATE_FRAMEWORK})
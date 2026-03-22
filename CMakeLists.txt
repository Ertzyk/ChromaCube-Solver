cmake_minimum_required(VERSION 3.15)
project(AgnosticCubeSolver)

# 1. Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 2. Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# 3. Find OpenCV
find_package(OpenCV REQUIRED)

# 4. Create the Python module named "cube_vision"
pybind11_add_module(cube_vision src/cpp/vision_bridge.cpp)

# 5. Link OpenCV
target_link_libraries(cube_vision PRIVATE ${OpenCV_LIBS})
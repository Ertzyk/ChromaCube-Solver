#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

namespace py = pybind11;

// Helper function to get the average HSV color of a specific Region of Interest
std::vector<int> get_average_hsv(const cv::Mat& frame, cv::Rect roi) {
    cv::Mat roi_bgr = frame(roi);
    cv::Mat roi_hsv;
    
    // Convert just this small square from BGR to HSV
    cv::cvtColor(roi_bgr, roi_hsv, cv::COLOR_BGR2HSV);
    
    // Calculate the mean color to ignore slight glares or shadows
    cv::Scalar avg = cv::mean(roi_hsv);
    
    return { static_cast<int>(avg[0]), static_cast<int>(avg[1]), static_cast<int>(avg[2]) };
}

std::vector<std::vector<int>> extract_hsv_colors() {
    cv::VideoCapture cap(0); // Open default webcam

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    if (!cap.isOpened()) {
        throw std::runtime_error("Error: Could not open the webcam.");
    }

    std::vector<std::vector<int>> all_facelets;
    cv::Mat frame;
    
    // Grid configuration
    int square_size = 80; 
    int gap = 16;    
    int faces_captured = 0;

    std::cout << "Webcam opened. Align the cube and press SPACE to capture a face. Press ESC to quit." << std::endl;

    while (faces_captured < 6) {
        cap >> frame;
        if (frame.empty()) break;

        // Flip frame horizontally for a more natural "mirror" feel
        cv::flip(frame, frame, 1); 

        // Calculate top-left corner to center the 3x3 grid
        int start_x = (frame.cols - (3 * square_size + 2 * gap)) / 2;
        int start_y = (frame.rows - (3 * square_size + 2 * gap)) / 2;

        std::vector<cv::Rect> current_grid;

        // Draw the 3x3 grid
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                int x = start_x + col * (square_size + gap);
                int y = start_y + row * (square_size + gap);
                cv::Rect roi(x, y, square_size, square_size);
                current_grid.push_back(roi);
                
                // Draw the white outline for the user to aim
                cv::rectangle(frame, roi, cv::Scalar(255, 255, 255), 2);

                // Calculate the real-time average BGR color of this specific square
                cv::Mat roi_bgr = frame(roi);
                cv::Scalar avg_bgr = cv::mean(roi_bgr);

                // Draw a small filled "color swatch" in the corner of the square
                cv::Rect swatch(x + 2, y + 2, 15, 15);
                cv::rectangle(frame, swatch, avg_bgr, cv::FILLED);
            }
        }

        // Add a UI text overlay
        std::string instructions = "Captured " + std::to_string(faces_captured) + "/6 faces. Press SPACE to scan.";
        cv::putText(frame, instructions, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Cube Scanner", frame);

        char key = (char)cv::waitKey(1);

        if (cv::getWindowProperty("Cube Scanner", cv::WND_PROP_VISIBLE) < 1) {
            std::cout << "Window closed by user." << std::endl;
            break; 
        }

        // Handle normal keyboard inputs
        if (key == 27) { // ESC key pressed
            break;
        } 
        else if (key == ' ') { // Spacebar pressed
            std::cout << "Captured face " << (faces_captured + 1) << "!" << std::endl;
            for (const auto& roi : current_grid) {
                all_facelets.push_back(get_average_hsv(frame, roi));
            }
            faces_captured++;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    
    return all_facelets;
}

PYBIND11_MODULE(cube_vision, m) {
    m.doc() = "C++ OpenCV bridge for Agnostic Cube Solver";
    m.def("extract_hsv_colors", &extract_hsv_colors, "Opens webcam, scans 6 faces, returns 54 HSV values");
}
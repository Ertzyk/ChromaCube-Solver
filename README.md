# ChromaCube Solver

*A real-time, color-agnostic Rubik's Cube solver that uses Computer Vision and Constrained Machine Learning to dynamically understand cube states in unpredictable environments.*

## The Problem
Most open-source Rubik's Cube solvers rely on hardcoded HSV thresholds (e.g., `if hue > 150: return RED`). This approach is incredibly fragile. It instantly breaks if the user has a cube with non-standard sticker shades, or if they scan the cube in a room with warm lighting, shadows, or screen glare. 

ChromaCube solves this by throwing out hardcoded color rules. Instead, it reads the physical environment in real-time and uses mathematical clustering to figure out what the colors *should* be.

## The Architecture
This project bridges a high-speed C++ vision frontend with a heavy mathematical Python backend using `pybind11`. 

### Phase 1: High-Speed Vision (C++ & OpenCV)
* Requests a 720p HD webcam feed and overlays a mathematically centered 340x340 pixel targeting grid.
* Captures the average HSV values of 54 facelets across 6 faces.
* Features a real-time "Visual Debugger" that draws live color swatches on the screen so the user can see exactly how the camera's sensor is interpreting the light.

### Phase 2: Decoupled Cartesian Transformation (Python & NumPy)
* Standard K-Means clustering fails on raw HSV data because Hue is a cylinder, causing shadows to mathematically pull identical colors apart.
* To fix this, the 54 cylindrical HSV arrays are projected into a custom 4D Cartesian space:
  * **X / Y:** Mapped to a fixed-radius Hue circle (forcing the algorithm to respect true color angles regardless of glare).
  * **Z:** Mapped strictly to Saturation (isolating the White face).
  * **W:** Mapped to a crushed Value axis to completely ignore environmental shadows.

### Phase 3: Constrained Clustering (SciPy)
* **Centroid Seeding:** Standard K-Means random initialization often merges Red and Orange. We intercept the algorithm and mathematically anchor the starting clusters to the 6 physical center pieces of the cube.
* **The Hungarian Algorithm:** To prevent lighting anomalies from tricking the AI into assigning 10 pieces to one color, the pipeline uses `scipy.optimize.linear_sum_assignment`. This guarantees a strict 9-to-9 mathematical distribution of facelets to their most optimal color clusters.

### Phase 4: Algorithmic Resolution
* The clustered labels are mathematically mirrored (to correct for webcam hardware flipping) and re-ordered into a strict U-R-F-D-L-B string.
* The string is fed into Kociemba's Two-Phase algorithm to calculate a near-optimal solution (usually 20 moves or less) almost instantly.

## Installation & Usage

### Prerequisites
* **C++ Compiler:** MSVC (Windows) or GCC/Clang (Mac/Linux).
* **CMake & Ninja:** For building the C++ bridge.
* **OpenCV:** Installed and built on your system.
* **Python 3.8+** 

### Build the C++ Vision Engine
Clone the repository and build the `pybind11` module. *(Note: Update the CMake flags to point to your local Python executable and OpenCV build directories).*
```bash
mkdir build
cd build
cmake -G Ninja -DPython_EXECUTABLE="path/to/your/python.exe" -DOpenCV_DIR="path/to/your/opencv/build" ..
ninja
```

### Install Python Dependencies
```bash
cd ..
pip install -r requirements.txt
```

### Run the Solver
```bash
python src/python/main.py
```

### Scanning Instructions
Start by holding the cube in a neutral position (Front face pointing at the camera, Top face pointing at the ceiling).
1. **UP:** Tilt the top of the cube towards the camera to scan the Top face.
2. **FRONT:** Tilt the cube back to the neutral position to scan the Front face.
3. **RIGHT:** Rotate the entire cube to the left.
4. **BACK:** Rotate the cube to the left again.
5. **LEFT:** Rotate the cube to the left again.
6. **DOWN:** Rotate the cube left one last time (back to neutral), then tilt the bottom of the cube towards the camera.
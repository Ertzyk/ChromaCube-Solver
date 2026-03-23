import sys
import os

sys.path.append(os.path.abspath('./build')) 
opencv_bin_path = r"C:\opencv\opencv\build\x64\vc16\bin"
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(opencv_bin_path)

import cube_vision
from clustering import hsv_to_kociemba_string
from solver import solve_cube

def print_cube_net(cube_string):
    """
    Takes the 54-char string and prints it as a 2D unfolded cross for debugging.
    """
    print("\n--- KOCIEMBA 2D NET MAP ---")
    U = cube_string[0:9]
    R = cube_string[9:18]
    F = cube_string[18:27]
    D = cube_string[27:36]
    L = cube_string[36:45]
    B = cube_string[45:54]

    # Print Up face
    for i in range(0, 9, 3):
        print(f"       {U[i]} {U[i+1]} {U[i+2]}")
        
    # Print Left, Front, Right, Back faces inline
    for i in range(0, 9, 3):
        print(f"{L[i]} {L[i+1]} {L[i+2]}  {F[i]} {F[i+1]} {F[i+2]}  {R[i]} {R[i+1]} {R[i+2]}  {B[i]} {B[i+1]} {B[i+2]}")
        
    # Print Down face
    for i in range(0, 9, 3):
        print(f"       {D[i]} {D[i+1]} {D[i+2]}")
    print("---------------------------\n")

def main():
    print("=========================================")
    print("         CUBE SOLVER         ")
    print("=========================================\n")
    print("Scan the faces in this exact physical sequence:")
    print("1. UP    (Top face)")
    print("2. FRONT (Tilt the cube DOWN)")
    print("3. RIGHT (Rotate the cube LEFT)")
    print("4. BACK  (Rotate the cube LEFT)")
    print("5. LEFT  (Rotate the cube LEFT)")
    print("6. DOWN  (Rotate LEFT to Front, then tilt UP)")
    print("\nOpening C++ Vision Scanner...")
    
    colors = cube_vision.extract_hsv_colors()
    
    if len(colors) == 54:
        print("\n[+] Vision pipeline finished. 54 facelets extracted.")
        print("[*] Running K-Means ML clustering...")
        
        try:
            kociemba_string = hsv_to_kociemba_string(colors)
            print("[+] Clustering successful!")
            
            print_cube_net(kociemba_string)
            
            print("[*] Calculating optimal solution path...")
            solution = solve_cube(kociemba_string)
            
            print("\n=========================================")
            print("             SOLUTION FOUND!             ")
            print("=========================================")
            print(f"Moves: {solution}")
            print("=========================================")
            
        except Exception as e:
            print(f"\n[!] Pipeline Error: {e}")
    else:
        print("\n[!] Scanner closed before all 6 faces were captured.")

if __name__ == "__main__":
    main()
import kociemba

def solve_cube(cube_string):
    # Takes a 54-character string of U, R, F, D, L, B and calculates the winning moves.
    try:
        solution = kociemba.solve(cube_string)
        return solution
    except ValueError as e:
        raise ValueError("Invalid cube state. The scan was likely inaccurate or out of order. Please try scanning again.")
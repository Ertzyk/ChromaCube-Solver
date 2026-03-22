import kociemba

def solve_cube(cube_string):
    """
    Takes a 54-character string of U, R, F, D, L, B and calculates the winning moves.
    """
    try:
        solution = kociemba.solve(cube_string)
        return solution
        
    except ValueError as e:
        # If the K-Means clustering failed to accurately map the colors, or if the user
        # scanned a face twice, the resulting string will be mathematically impossible 
        # to solve. Kociemba will throw a ValueError.
        raise ValueError("Invalid cube state. The scan was likely inaccurate or out of order. Please try scanning again.")
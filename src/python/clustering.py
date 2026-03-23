import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def hsv_to_cartesian(hsv_array):
    """Converts OpenCV HSV to Cartesian by decoupling Hue from Saturation."""
    hsv_array = np.array(hsv_array, dtype=float)
    H = hsv_array[:, 0]
    S = hsv_array[:, 1]
    V = hsv_array[:, 2]
    H_rad = H * (np.pi / 90.0)
    
    # THE HUE CIRCLE (X, Y)
    X = np.cos(H_rad) * 100.0
    Y = np.sin(H_rad) * 100.0
    
    # THE SATURATION AXIS (Z)
    Z = S 
    
    # We completely ignore brightness except for a tiny fraction to break mathematical ties.
    W = V * 0.1
    
    # Return a 4D feature space for clustering
    return np.column_stack((X, Y, Z, W))

def hsv_to_kociemba_string(hsv_facelets):
    """
    Takes 54 HSV arrays, converts to Cartesian space, clusters them, 
    and maps them to U, R, F, D, L, B.
    """
    if len(hsv_facelets) != 54:
        raise ValueError(f"Expected 54 facelets, but received {len(hsv_facelets)}.")
    
    check_environmental_lighting(hsv_facelets)

    # Convert to NumPy array and transform the feature space
    X_cylindrical = np.array(hsv_facelets)
    X_cartesian = hsv_to_cartesian(X_cylindrical)

    center_indices = [4, 13, 22, 31, 40, 49]
    initial_centers = X_cartesian[center_indices]

    # Run K-Means to find the perfect environmental color profiles
    kmeans = KMeans(n_clusters = 6, init = initial_centers, n_init = 1, random_state = 42)
    kmeans.fit(X_cartesian) # We fit, but we DO NOT trust its default predict()

    # Create exactly 54 slots (9 slots for each of the 6 K-Means centers)
    target_centroids = np.repeat(kmeans.cluster_centers_, 9, axis = 0)
    
    # Calculate the distance from every single facelet to every available slot
    cost_matrix = cdist(X_cartesian, target_centroids, metric = 'euclidean')
    
    # Force the optimal 1-to-1 mathematical assignment
    _, col_ind = linear_sum_assignment(cost_matrix)
    
    # Map the 54 matched slots back to the 6 cluster IDs
    labels = col_ind // 9

    # Map cluster IDs to standard Kociemba faces (Centers are at indices 4, 13, 22, 31, 40, 49)
    centers_map = {
        labels[4]:  'U',
        labels[13]: 'F',
        labels[22]: 'R',
        labels[31]: 'B',
        labels[40]: 'L',
        labels[49]: 'D'
    }

    if len(centers_map) != 6:
        print("\n--- DEBUG: CENTER COLOR COLLISION DETECTED ---")
        print("K-Means assigned two or more center pieces to the same color cluster.")
        print("Here is what the webcam actually saw for the 6 centers:")
        
        center_names = {4: 'U (Top)', 13: 'F (Front)', 22: 'R (Right)', 31: 'B (Back)', 40: 'L (Left)', 49: 'D (Down)'}
        
        for idx, name in center_names.items():
            raw_hsv = hsv_facelets[idx]
            cluster_id = labels[idx]
            print(f"Face {name:<10} | HSV: {raw_hsv[0]:>3}, {raw_hsv[1]:>3}, {raw_hsv[2]:>3} | Cluster ID: {cluster_id}")
            
        print("----------------------------------------------\n")
        raise ValueError("Error: K-Means merged centers")

    def mirror_face(face_array):
        return face_array.reshape(3, 3)[:, ::-1].flatten()

    # Reconstruct the array, applying the mirror fix to every face
    u_face = mirror_face(labels[0:9])
    f_face = mirror_face(labels[9:18])
    r_face = mirror_face(labels[18:27])
    b_face = mirror_face(labels[27:36])
    l_face = mirror_face(labels[36:45])
    d_face = mirror_face(labels[45:54])

    kociemba_ordered_labels = np.concatenate([u_face, r_face, f_face, d_face, l_face, b_face])

    # Build the final string
    cube_string = "".join([centers_map[label] for label in kociemba_ordered_labels])

    return cube_string


def check_environmental_lighting(hsv_facelets):
    """
    Analyzes the Saturation of the 6 center pieces to ensure 
    the camera is actually seeing color, not just gray shadows.
    """
    # Extract the 6 center pieces
    centers = [hsv_facelets[i] for i in [4, 13, 22, 31, 40, 49]]
    saturations = [c[1] for c in centers]
    
    # Sort from lowest to highest saturation
    saturations.sort()
    
    # The lowest saturation is the White face (should be near 0). We ignore White and average the remaining 5 colored faces.
    colored_saturations = saturations[1:]
    avg_sat = sum(colored_saturations) / len(colored_saturations)
    
    # In a well-lit room, avg_sat should be 110+. 
    if avg_sat < 110:
        raise ValueError(
            f"Environment too dark! Average color saturation is critically low ({avg_sat:.1f}/255). "
            "Please turn on a room light and scan again."
        )
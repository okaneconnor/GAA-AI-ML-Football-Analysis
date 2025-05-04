# football_utils/team_assignment.py

import cv2
import numpy as np
from sklearn.cluster import KMeans
import traceback

def get_dominant_color(roi):
    """
    Calculates the dominant color in a Region of Interest (ROI)
    using KMeans clustering on pixels.
    Excludes very dark/bright pixels and potentially green (pitch color).
    """
    try:
        # Reshape ROI to a list of pixels
        pixels = roi.reshape(-1, 3).astype(np.float32)

        # --- Basic Filtering ---
        # Filter out black/very dark pixels (shadows, etc.)
        dark_threshold = 30
        mask_dark = np.all(pixels > dark_threshold, axis=1)
        # Filter out white/very bright pixels (highlights, lines)
        bright_threshold = 225
        mask_bright = np.all(pixels < bright_threshold, axis=1)

        # Filter out green pixels (pitch) - adjust thresholds as needed
        # Assuming BGR color space from OpenCV
        green_threshold = 60 # Min green value
        non_green_max = 50  # Max R and B value for it to be considered primarily green
        mask_green = ~((pixels[:, 1] > green_threshold) & (pixels[:, 0] < non_green_max) & (pixels[:, 2] < non_green_max))

        # Combine masks
        valid_mask = mask_dark & mask_bright & mask_green
        filtered_pixels = pixels[valid_mask]

        if len(filtered_pixels) < 20: # Need enough pixels to cluster reliably
             # Fallback: Use average color if filtering removes too much
             if len(pixels) > 0:
                 avg_color = np.mean(pixels, axis=0).astype(int)
                 # print(f"Warning: Too few pixels after filtering ({len(filtered_pixels)}), using average color {avg_color}")
                 return tuple(avg_color)
             else:
                 # print("Warning: No pixels in ROI.")
                 return None # Cannot determine color

        # --- KMeans Clustering ---
        n_clusters = 3 # Find a few dominant colors
        if len(filtered_pixels) < n_clusters:
             n_clusters = len(filtered_pixels) # Adjust if fewer pixels than clusters

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=100) # Use n_init='auto' in newer sklearn
        kmeans.fit(filtered_pixels)

        # Find the largest cluster (most frequent color)
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_label = unique[counts.argmax()]
        dominant_color = kmeans.cluster_centers_[dominant_label].astype(int)

        return tuple(dominant_color) # Return as BGR tuple

    except Exception as e:
        print(f"Error in get_dominant_color: {e}")
        # print(traceback.format_exc())
        # Fallback to average color on error if possible
        if 'pixels' in locals() and len(pixels) > 0:
             avg_color = np.mean(pixels, axis=0).astype(int)
             print(f"Using average color {avg_color} due to error.")
             return tuple(avg_color)
        return None


def assign_teams(frame, player_detections):
    """
    Assigns players to two teams based on dominant jersey color using KMeans.
    Args:
        frame: The current video frame.
        player_detections: A list of dictionaries, where each dict corresponds
                           to a detected player (must include 'xmin', 'ymin', 'xmax', 'ymax').
                           It's assumed this list *only* contains players.
    Returns:
        A dictionary mapping the original index of a player detection to a team ID (0 or 1).
        Returns an empty dictionary if teams cannot be assigned.
    """
    player_colors = {} # Store dominant color for each player index
    player_features = [] # List to store colors for clustering

    if not player_detections:
        return {}

    height, width, _ = frame.shape

    for idx, det in enumerate(player_detections):
        # Extract player ROI (Region of Interest) - focus on torso?
        xmin, ymin, xmax, ymax = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
        # Basic check for valid coordinates
        if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0 or xmax > width or ymax > height:
             print(f"Warning: Invalid bbox coordinates for detection index {idx}, skipping team assignment.")
             continue

        # --- Refine ROI to likely capture jersey ---
        box_h = ymax - ymin
        # Take middle part vertically (e.g., 25% down to 75% down)
        roi_ymin = ymin + int(box_h * 0.25)
        roi_ymax = ymax - int(box_h * 0.25)
        # Ensure roi dimensions are valid
        if roi_ymin >= roi_ymax or xmin >= xmax:
            roi = frame[ymin:ymax, xmin:xmax] # Fallback to full box if refined ROI is invalid
        else:
            roi = frame[roi_ymin:roi_ymax, xmin:xmax]
        # ------------------------------------------

        if roi.size == 0:
            # print(f"Warning: Empty ROI for detection index {idx}, skipping team assignment.")
            continue

        dominant_color = get_dominant_color(roi)

        if dominant_color:
            player_colors[idx] = dominant_color
            player_features.append(dominant_color) # Use BGR color as feature

    if len(player_features) < 2:
        print("Warning: Not enough players with identifiable colors to form two teams.")
        return {} # Cannot cluster less than 2 players

    # Cluster the extracted dominant colors into 2 teams
    try:
        kmeans_teams = KMeans(n_clusters=2, random_state=0, n_init=10).fit(np.array(player_features))
        team_assignments = {}
        player_indices = list(player_colors.keys()) # Get indices corresponding to features

        for i in range(len(player_features)):
            original_player_idx = player_indices[i]
            team_id = kmeans_teams.labels_[i]
            team_assignments[original_player_idx] = team_id # Map original index to team ID

        # print(f"Team assignments: {team_assignments}") # Debug print
        return team_assignments

    except Exception as e:
        print(f"Error during KMeans clustering for team assignment: {e}")
        print(traceback.format_exc())
        return {} # Return empty on clustering error
def is_in_field_region(box, frame_height, frame_width, field_region=None):
    """Check if detection is in the main playing field."""
    if field_region is None:
        # Default field region if not specified
        field_region = {
            'top': 0.45,        # Vertical position below fence
            'bottom': 0.90,     # Bottom of field
            'left': 0.05,       # Left edge with small margin
            'right': 0.95       # Right edge with small margin
        }
    
    x1, y1, x2, y2 = box
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2
    
    in_field = (
        box_center_y > field_region['top'] * frame_height and
        box_center_y < field_region['bottom'] * frame_height and
        box_center_x > field_region['left'] * frame_width and
        box_center_x < field_region['right'] * frame_width
    )
    
    return in_field

def has_player_movement_pattern(track_history, min_window=5):
    """Check if movement pattern is consistent with a player."""
    if not track_history or len(track_history) < min_window:
        return True  # Not enough data
    
    # Calculate velocities between consecutive points
    velocities = []
    track_list = list(track_history)
    for i in range(1, len(track_list)):
        p1 = np.array(track_list[i-1])
        p2 = np.array(track_list[i])
        velocity = np.sqrt(np.sum((p2-p1)**2))
        velocities.append(velocity)
    
    # Player movement patterns: not too static, not erratic
    avg_velocity = np.mean(velocities)
    velocity_std = np.std(velocities)
    
    # More specific player movement pattern
    return 0.3 < avg_velocity < 35 and velocity_std < 18

def assign_teams(frame, detections, visualize=False):
    """Main function to assign players to teams using a robust multi-method approach."""
    # Initialize containers for player analysis
    player_info = []

    # Pre-process all player detections
    for i, det in enumerate(detections):
        if det['name'] != 'person':
            continue

        # Extract jersey ROI
        jersey_roi = extract_jersey_roi(frame, det)
        if jersey_roi is None:
            continue

        # Check if player is a referee (black jersey)
        if is_referee(jersey_roi):
            continue

        # Filter out green field background
        filtered_roi = filter_green_field(jersey_roi)
        if filtered_roi is None:
            continue

        # Get dominant colors
        dominant_colors = detect_dominant_colors(filtered_roi)
        if not dominant_colors:
            continue

        # Analyze color distribution to get the most representative jersey color
        jersey_color = analyze_color_distribution(dominant_colors)
        if jersey_color is None:
            continue

        # Calculate RGB-based color score
        rgb_score = get_red_vs_blue_score(jersey_color)

        # Multi-method team classification
        is_red = False
        is_blue = False

        # Method 1: HSV-based detection (most reliable)
        if is_red_jersey_hsv(filtered_roi):
            is_red = True
        elif is_blue_jersey_hsv(filtered_roi):
            is_blue = True
        # Method 2: Check RGB channels
        elif jersey_color[2] > max(jersey_color[0], jersey_color[1])*1.5 and jersey_color[2] > 100:  # Red is dominant
            is_red = True
        elif jersey_color[0] > max(jersey_color[1], jersey_color[2])*1.2 and jersey_color[0] > 100:  # Blue is dominant
            is_blue = True
        # Method 3: RGB difference score as fallback
        elif rgb_score > 30:  # Significant red dominance
            is_red = True
        elif rgb_score < -10:  # Blue dominance
            is_blue = True

        # Store player information
        player_info.append({
            'id': i,
            'detection': det,
            'jersey_color': jersey_color,
            'is_red': is_red,
            'is_blue': is_blue,
            'rgb_score': rgb_score,
            'jersey_roi': jersey_roi,
            'filtered_roi': filtered_roi
        })

    # Handle edge cases and uncertain classifications
    # First, identify confident classifications
    confident_red = [p for p in player_info if p['is_red'] and not p['is_blue']]
    confident_blue = [p for p in player_info if p['is_blue'] and not p['is_red']]
    uncertain = [p for p in player_info if not p['is_red'] and not p['is_blue']]

    # Use confident classifications to guide uncertain ones
    if uncertain and (confident_red or confident_blue):
        if confident_red:
            avg_red_color = np.mean([p['jersey_color'] for p in confident_red], axis=0)
        if confident_blue:
            avg_blue_color = np.mean([p['jersey_color'] for p in confident_blue], axis=0)

        for player in uncertain:
            color = player['jersey_color']
            if confident_red and confident_blue:
                # Compare distance to both team colors
                dist_to_red = np.linalg.norm(color - avg_red_color)
                dist_to_blue = np.linalg.norm(color - avg_blue_color)

                if dist_to_red < dist_to_blue:
                    player['is_red'] = True
                else:
                    player['is_blue'] = True
            elif confident_red:
                # Check if similar to red
                if color[2] > 80 or get_red_vs_blue_score(color) > 0:
                    player['is_red'] = True
                else:
                    player['is_blue'] = True
            elif confident_blue:
                # Check if similar to blue
                if color[0] > 80 or get_red_vs_blue_score(color) < 0:
                    player['is_blue'] = True
                else:
                    player['is_red'] = True

    # Make final assignments
    team_assignments = {}
    team_colors = []

    # Separate players into teams
    red_players = [p for p in player_info if p['is_red'] or (not p['is_blue'] and p['rgb_score'] > 0)]
    blue_players = [p for p in player_info if p['is_blue'] or (not p['is_red'] and p['rgb_score'] <= 0)]

    # Calculate team colors
    if red_players:
        red_jersey_colors = [p['jersey_color'] for p in red_players]
        avg_red = np.median(red_jersey_colors, axis=0).astype(int)
        # Enhance red component for visualization
        avg_red = np.array([avg_red[0], avg_red[1], min(255, int(avg_red[2] * 1.2))])
        team_colors.append(avg_red)
    else:
        team_colors.append(np.array([0, 0, 200]))  # Default red if none found

    if blue_players:
        blue_jersey_colors = [p['jersey_color'] for p in blue_players]
        avg_blue = np.median(blue_jersey_colors, axis=0).astype(int)
        # Enhance blue component for visualization
        avg_blue = np.array([min(255, int(avg_blue[0] * 1.2)), avg_blue[1], avg_blue[2]])
        team_colors.append(avg_blue)
    else:
        team_colors.append(np.array([200, 0, 0]))  # Default blue if none found

    # Assign team IDs (0 for blue, 1 for red)
    for p in red_players:
        team_assignments[p['id']] = 1
    for p in blue_players:
        team_assignments[p['id']] = 0

    # Convert to our expected format for compatibility
    final_assignments = {}
    for player in player_info:
        if player['id'] in team_assignments:
            det_idx = detections.index(player['detection'])
            final_assignments[det_idx] = team_assignments[player['id']]

    return final_assignments, team_colorsdef analyze_color_distribution(colors):
    """Analyze the distribution of colors to find the most representative color."""
    if not colors or len(colors) == 0:
        return None

    # Start with the most dominant color
    primary_color, primary_count = colors[0]

    # Check if we have a second color that's close in frequency
    if len(colors) > 1:
        secondary_color, secondary_count = colors[1]
        primary_percentage = primary_count / (primary_count + secondary_count)

        # If the top two colors are close in frequency, check which is more saturated
        if primary_percentage < 0.65:  # Less than 65% dominance
            primary_min = min(primary_color)
            primary_max = max(primary_color)
            primary_saturation = 0 if primary_max == 0 else (primary_max - primary_min) / primary_max

            secondary_min = min(secondary_color)
            secondary_max = max(secondary_color)
            secondary_saturation = 0 if secondary_max == 0 else (secondary_max - secondary_min) / secondary_max

            # If the secondary color is more saturated, and it's a team color, use it
            if secondary_saturation > primary_saturation * 1.5:
                b, g, r = secondary_color
                if (r > g * 1.5 and r > b * 1.5) or (b > r * 1.2 and b > g * 1.2):
                    return secondary_color

    return primary_color

def is_red_jersey_hsv(roi):
    """Detect red jerseys using HSV color space which is better for color detection."""
    if roi is None or roi.size == 0:
        return False

    # Convert to HSV for better color detection
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define two masks for red (red hue wraps around in HSV)
    lower_red1 = np.array([0, 100, 60])  # Lower range for red
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)

    lower_red2 = np.array([170, 100, 60])  # Upper range for red (wraps around)
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)

    # Combine masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Calculate percentage of red pixels
    total_pixels = roi.shape[0] * roi.shape[1]
    red_pixels = cv2.countNonZero(mask)
    red_percentage = (red_pixels / total_pixels) * 100

    # Return True if percentage exceeds threshold
    return red_percentage > 25  # 25% of pixels need to be red

def is_blue_jersey_hsv(roi):
    """Detect blue jerseys using HSV color space."""
    if roi is None or roi.size == 0:
        return False

    # Convert to HSV for better color detection
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define mask for blue
    lower_blue = np.array([100, 50, 50])  # Blue hue range with moderate saturation
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)

    # Calculate percentage of blue pixels
    total_pixels = roi.shape[0] * roi.shape[1]
    blue_pixels = cv2.countNonZero(mask)
    blue_percentage = (blue_pixels / total_pixels) * 100

    # Return True if percentage exceeds threshold
    return blue_percentage > 20  # 20% of pixels need to be blue

def get_red_vs_blue_score(color):
    """Calculate a score representing how red vs blue a color is.
    Positive values indicate more red, negative values indicate more blue."""
    b, g, r = color  # BGR format

    # Calculate red dominance over blue with more weight on red
    red_blue_diff = int(r) - int(b)

    # Weight the red component more heavily (tuned for GAA red jerseys)
    if r > 100 and r > 1.5 * b:
        red_blue_diff += 30

    # Weight the blue component more heavily (tuned for GAA blue jerseys)
    if b > 80 and b > 1.2 * r:
        red_blue_diff -= 30

    return red_blue_diffdef filter_green_field(roi):
    """Remove green field colors from ROI"""
    if roi is None:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define a range for green field colors
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])

    # Create a mask and apply it
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(mask)  # Invert to get non-green areas

    # Apply mask to ROI
    filtered = cv2.bitwise_and(roi, roi, mask=mask)
    return filtered

def is_referee(roi):
    """Detect referee based on black/dark uniform."""
    if roi is None or roi.size < 15: 
        return False
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Lower Max Value (brightness) threshold for very dark uniforms
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 50])  # Very dark threshold
    
    dark_mask = cv2.inRange(hsv_roi, lower_dark, upper_dark)
    dark_ratio = cv2.countNonZero(dark_mask) / (roi.shape[0] * roi.shape[1])
    
    # Return True if enough dark pixels (ratio > 70%)
    return dark_ratio > 0.70

def detect_dominant_colors(roi, k=3):
    """Extract dominant colors using K-means clustering."""
    if roi is None or roi.size == 0 or roi.size < 100*3:  # Minimum size check
        return None

    # Reshape for clustering
    pixels = roi.reshape(-1, 3)

    # Filter out very dark and very light pixels that might be noise
    mask = np.all((pixels > [15, 15, 15]) & (pixels < [240, 240, 240]), axis=1)
    filtered_pixels = pixels[mask]

    # If not enough pixels left after filtering, use original
    if len(filtered_pixels) < 100:
        filtered_pixels = pixels

    # Run k-means with multiple initializations for better results
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(filtered_pixels)

    # Get colors and counts
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)

    # Sort by frequency
    color_counts = [(colors[i], counts[i]) for i in range(len(counts))]
    color_counts.sort(key=lambda x: x[1], reverse=True)

    return color_counts# football_utils/team_assignment.py

import cv2
import numpy as np
from sklearn.cluster import KMeans
import traceback
from collections import defaultdict, deque

# --- Configuration ---
MIN_PLAYER_HEIGHT_PIXELS = 40  # Minimum height for players

def extract_jersey_roi(frame, detection):
    """Extract jersey region focusing on the torso area where team colors are most prominent"""
    if isinstance(detection, dict):
        x1, y1, x2, y2 = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
    else:
        x1, y1, x2, y2 = map(int, detection)

    # Focus on upper body (jersey area)
    height = y2 - y1
    width = x2 - x1
    
    if height < MIN_PLAYER_HEIGHT_PIXELS or width < 5:
        return None

    # Calculate torso region with tighter focus on jersey area
    torso_y1 = int(y1 + height * 0.15)  # Skip head
    torso_y2 = int(y1 + height * 0.45)  # Upper torso only
    torso_x1 = int(x1 + width * 0.20)
    torso_x2 = int(x2 - width * 0.20)

    # Ensure coordinates are within image bounds
    frame_h, frame_w = frame.shape[:2]
    torso_y1 = max(0, min(torso_y1, frame_h - 1))
    torso_y2 = max(torso_y1 + 5, min(torso_y2, frame_h - 1))
    torso_x1 = max(0, min(torso_x1, frame_w - 1))
    torso_x2 = max(torso_x1 + 3, min(torso_x2, frame_w - 1))

    # Skip invalid regions
    if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
        return None

    # Extract torso ROI
    try:
        jersey_roi = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if jersey_roi.size == 0:
            return None
        return jersey_roi if jersey_roi.size >= 30 else None
    except Exception as e:
        print(f"Error extracting jersey: {e}")
        return None
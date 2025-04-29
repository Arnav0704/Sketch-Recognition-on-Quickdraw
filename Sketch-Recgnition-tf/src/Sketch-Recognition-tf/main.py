import os
import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
import threading
import time 

# --- Configuration --- 
MODEL_PATH = "/mnt/d/model/quick_draw/LSTM/best_stroke_lstm_model_final.keras"  # Path to your saved model
MAX_LEN = 196
CLASSES = [
    "apple", "banana", "book", "car", "cat", "chair", "cloud", "dog", "door", "eye",
    "face", "fish", "flower", "fork", "guitar", "hammer", "hat", "house", "key", "knife",
    "leaf", "lightning", "moon", "mountain", "mouse", "star", "sun", "table", "tree", "umbrella"
]
NUM_CLASSES = len(CLASSES)

# IMPORTANT: Use the exact values printed during your training!
MEAN_VALS = np.array([3.0792246, 2.8368876], dtype=np.float32)
STD_VALS = np.array([36.755478, 37.947872], dtype=np.float32)

# Add a small epsilon to std dev to prevent division by zero
STD_VALS = np.where(STD_VALS == 0, 1e-6, STD_VALS)

# --- Pygame Setup ---
pygame.init()
width, height = 900, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Sketch Recognition App")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PREDICTION_INTERVAL = 0.01  # Seconds between predictions
last_prediction_time = 0
prediction_trigger = False

# Button dimensions
button_width = 150
button_height = 50
margin = 20
# predict_button = pygame.Rect(width - button_width - margin, margin, button_width, button_height)
clear_button = pygame.Rect(width - button_width - margin - 40, button_height, button_width, button_height)

# Drawing canvas area - Make it square to match QuickDraw format
canvas_size = min(width - (2*margin + button_width + margin), height - 2*margin)
canvas_rect = pygame.Rect(margin, margin, canvas_size, canvas_size)

# Font setup
pygame.font.init()
font = pygame.font.SysFont('Arial', 24)
small_font = pygame.font.SysFont('Arial', 18)

# --- Drawing Variables ---
drawing = False
current_stroke = []  # Current stroke being drawn
all_strokes = []  # All completed strokes
predictions = []  # To store prediction results
predicting = False  # Flag for prediction status
model_loaded = False

# --- Model Loading ---
try:
    model = load_model(MODEL_PATH)
    model_loaded = True
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model_loaded = False
    print(f"Error loading model: {e}")
    print("The app will work, but predictions won't be available.")

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def simplify_stroke(stroke, epsilon=5.0):
    """
    Simplify a stroke using Ramer-Douglas-Peucker algorithm.
    
    Args:
        stroke: List of (x,y) points
        epsilon: Distance threshold for simplification
        
    Returns:
        Simplified list of points
    """
    if len(stroke) < 2:
        return stroke
    
    # Check if all points are identical
    all_same = all(p == stroke[0] for p in stroke)
    if all_same:
        return [stroke[0], stroke[0]]  # Return start and "end"
    
    # Find the point with the maximum distance from line between start and end
    max_dist = 0
    index = 0
    start, end = stroke[0], stroke[-1]
    
    # Special case: If start and end are the same point
    if start == end:
        # Find the farthest point from start/end
        for i in range(1, len(stroke) - 1):
            dist = distance(start, stroke[i])
            if dist > max_dist:
                max_dist = dist
                index = i
                
        if max_dist > epsilon:
            # Recursive calls
            left = simplify_stroke(stroke[:index+1], epsilon)
            right = simplify_stroke(stroke[index:], epsilon)
            # Merge the results (excluding duplicated point)
            return left[:-1] + right
        else:
            return [start, stroke[index]]
    
    # Regular case: Find maximum distance from line
    line_length = distance(start, end)
    if line_length == 0:  # Same start and end points
        return [start]
        
    for i in range(1, len(stroke) - 1):
        # Calculate distance from point to line
        p = stroke[i]
        # Line formula ax + by + c = 0
        a = end[1] - start[1]
        b = start[0] - end[0]
        c = end[0] * start[1] - start[0] * end[1]
        dist = abs(a * p[0] + b * p[1] + c) / math.sqrt(a**2 + b**2)
        
        if dist > max_dist:
            max_dist = dist
            index = i
    
    # If max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon:
        # Recursive calls
        left = simplify_stroke(stroke[:index+1], epsilon)
        right = simplify_stroke(stroke[index:], epsilon)
        # Merge the results (excluding duplicated point)
        return left[:-1] + right
    else:
        return [start, end]

def resample_stroke(stroke, target_points=20):
    """
    Resample a stroke to have a specific number of points.
    If stroke has fewer points than target, return the original.
    """
    if len(stroke) <= target_points:
        return stroke
    
    # Calculate total length with epsilon protection
    total_length = 0
    for i in range(1, len(stroke)):
        total_length += max(distance(stroke[i-1], stroke[i]), 1e-6)
    
    # Handle zero-length strokes
    if total_length < 1e-6:
        return [stroke[0]] * target_points
        
    # Calculate desired interval length
    interval_length = total_length / (target_points - 1)
    
    # Resample points
    resampled = [stroke[0]]  # Always keep the first point
    current_length = 0
    current_point = 0
    
    while current_point < len(stroke) - 1:
        segment_length = distance(stroke[current_point], stroke[current_point + 1])
        
        if current_length + segment_length >= interval_length:
            # Interpolate new point
            t = (interval_length - current_length) / segment_length
            new_x = stroke[current_point][0] + t * (stroke[current_point + 1][0] - stroke[current_point][0])
            new_y = stroke[current_point][1] + t * (stroke[current_point + 1][1] - stroke[current_point][1])
            resampled.append((new_x, new_y))
            
            # Reset for next interval but stay on same segment
            current_length = 0
            # Don't increment current_point
        else:
            current_length += segment_length
            current_point += 1
            
            # If we've reached the last point and still need more
            if current_point == len(stroke) - 1 and len(resampled) < target_points:
                resampled.append(stroke[-1])
    
    # Ensure we have the right number of points (may be off by 1 due to rounding)
    if len(resampled) < target_points:
        resampled.append(stroke[-1])
    
    return resampled[:target_points]

def preprocess_strokes(strokes):
    """
    Convert pygame strokes to QuickDraw format, simplify, and scale to 0-255 range.
    
    Args:
        strokes: List of pygame strokes, where each stroke is a list of (x,y) tuples
        
    Returns:
        List of strokes in QuickDraw format: [[[x0,x1,...],[y0,y1,...]], ...]
    """
    if not strokes:
        return []
    
    # Simplify strokes - reduce number of points
    simplified_strokes = []
    for stroke in strokes:
        if len(stroke) >= 3:
            # First simplify using RDP algorithm
            simplified = simplify_stroke(stroke, epsilon=5.0)
            # Then resample to have roughly 10-20 points per stroke
            resampled = resample_stroke(simplified, target_points=min(20, len(simplified)))
            simplified_strokes.append(resampled)
        else:
            simplified_strokes.append(stroke)
    
    # Find min/max values to normalize
    all_points = [point for stroke in simplified_strokes for point in stroke]
    if not all_points:
        return []
        
    min_x = min(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    
    # Shift to start at (0,0)
    shifted_strokes = []
    for stroke in simplified_strokes:
        shifted_stroke = [(x - min_x, y - min_y) for x, y in stroke]
        shifted_strokes.append(shifted_stroke)
    
    # Find max values after shifting
    all_shifted = [point for stroke in shifted_strokes for point in stroke]
    # Handle empty strokes after simplification
    if not all_shifted:
        return []
    
    # Calculate max dimensions with protection
    max_x = max(point[0] for point in all_shifted) or 1
    max_y = max(point[1] for point in all_shifted) or 1
    max_dim = max(max_x, max_y, 1)  # Ensure at least 1
    
    scaling_factor = 255.0 / max_dim
    
    # Convert to QuickDraw format
    quickdraw_strokes = []
    for stroke in shifted_strokes:
        if not stroke:
            continue
            
        x_coords = []
        y_coords = []
        
        for x, y in stroke:
            # Scale to 0-255 range
            scaled_x = int(x * scaling_factor)
            scaled_y = int(y * scaling_factor)
            
            x_coords.append(scaled_x)
            y_coords.append(scaled_y)
            
        if x_coords and y_coords:  # Only add non-empty strokes
            quickdraw_strokes.append([x_coords, y_coords])
    
    return quickdraw_strokes

def strokes_to_deltas(drawing_strokes):
    """
    Converts raw strokes list [ [[x,...],[y,...]], ...]
    to delta format [(dx, dy, pen_state), ...].
    pen_state = 0 for intermediate points, 1 for last point in stroke.
    """
    deltas = []
    last_x, last_y = 0, 0
    
    for stroke in drawing_strokes:
        x_coords, y_coords = stroke[0], stroke[1]
        if not x_coords:  # Skip empty strokes if any
            continue

        # First point uses absolute coords (or diff from 0,0)
        dx = x_coords[0] - last_x
        dy = y_coords[0] - last_y
        deltas.append([dx, dy, 0])  # pen_state=0 for first point

        # Subsequent points use deltas
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            deltas.append([dx, dy, 0])  # pen_state=0 for intermediate

        # Mark the last point of the stroke
        if deltas:  # Ensure deltas is not empty
            deltas[-1][2] = 1

        last_x, last_y = x_coords[-1], y_coords[-1]

    # Truncate if longer than MAX_LEN
    if len(deltas) > MAX_LEN:
        deltas = deltas[:MAX_LEN]

    return np.array(deltas, dtype=np.float32)

def preprocess_single_drawing(raw_strokes, max_len, mean_vals, std_vals):
    """Preprocesses a single drawing for prediction."""
    try:
        if not raw_strokes:
            return np.zeros((1, max_len, 3), dtype=np.float32)
            
        delta_sequence = strokes_to_deltas(raw_strokes)
        if delta_sequence.shape[0] < 2:
            return np.zeros((1, max_len, 3), dtype=np.float32)
        
        # Apply safe normalization
        delta_sequence[:, 0] = np.nan_to_num(
            (delta_sequence[:, 0] - mean_vals[0]) / std_vals[0],
            nan=0.0, posinf=0.0, neginf=0.0
        )
        delta_sequence[:, 1] = np.nan_to_num(
            (delta_sequence[:, 1] - mean_vals[1]) / std_vals[1],
            nan=0.0, posinf=0.0, neginf=0.0
        )
        
        padded_sequence = pad_sequences([delta_sequence], maxlen=max_len,
                                       padding='post', dtype='float32')
        return padded_sequence
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return np.zeros((1, max_len, 3), dtype=np.float32)

    return padded_sequence

def get_all_strokes_for_prediction():
    """Combine completed strokes and current stroke (if any) for prediction"""
    global all_strokes, current_stroke
    strokes = all_strokes.copy()
    if current_stroke and len(current_stroke) > 1:
        strokes.append(current_stroke.copy())
    return strokes

def predict_continuously():
    """Continuous prediction thread function"""
    global predictions, predicting, last_prediction_time
    while running:
        current_time = time.time()
        if current_time - last_prediction_time >= PREDICTION_INTERVAL:
            strokes = get_all_strokes_for_prediction()
            if strokes:
                try:
                    quickdraw_strokes = preprocess_strokes(strokes)
                    if quickdraw_strokes:
                        processed_input = preprocess_single_drawing(
                            quickdraw_strokes, MAX_LEN, MEAN_VALS, STD_VALS
                        )
                        pred = model.predict(processed_input, verbose=1)[0]
                        top_indices = pred.argsort()[-10:][::-1]
                        predictions = [(CLASSES[i], float(pred[i])) for i in top_indices]
                except Exception as e:
                    print(f"Prediction error: {e}")
                finally:
                    last_prediction_time = current_time
        time.sleep(0.2)  
        
# Start prediction thread
running = True
prediction_lock = threading.Lock()
model_lock = threading.Lock()  # Lock for model.predict
prediction_thread = threading.Thread(target=predict_continuously)
prediction_thread.daemon = True 
prediction_thread.start()

def clear_canvas():
    global all_strokes, current_stroke, predictions
    all_strokes = []
    current_stroke = []
    predictions = []
    
def main():
    global drawing, current_stroke, all_strokes, predictions, running
    
    while running:
        screen.fill(BLACK)
        
        # Draw the canvas border
        pygame.draw.rect(screen, GRAY, canvas_rect, 2)
        
        # Draw clear button only
        pygame.draw.rect(screen, RED, clear_button)
        clear_text = font.render('Clear', True, BLACK)
        clear_text_rect = clear_text.get_rect(center=clear_button.center)
        screen.blit(clear_text, clear_text_rect)
        
        # Draw existing strokes
        for stroke in all_strokes:
            if len(stroke) > 1:
                pygame.draw.lines(screen, WHITE, False, stroke, 3)
        
        # Draw current stroke
        if len(current_stroke) > 1:
            pygame.draw.lines(screen, WHITE, False, current_stroke, 3)
        
        current_predictions = []
        with prediction_lock:
            current_predictions = predictions.copy()

        if current_predictions:
            prediction_y = margin
            title_text = font.render('Live Predictions:', True, WHITE)
            screen.blit(title_text, (margin + 580, height - 450))
            
            for i, (class_name, probability) in enumerate(current_predictions):
                color = WHITE if i == 0 else GRAY
                pred_text = f"{i+1}. {class_name.capitalize()}: {probability:.6f}"
                pred_surface = font.render(pred_text, True, color)
                screen.blit(pred_surface, (margin + 580, height - 420 + (i * 30)))
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if canvas_rect.collidepoint(event.pos):
                    drawing = True
                    current_stroke = [event.pos]
                elif clear_button.collidepoint(event.pos):
                    clear_canvas()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if drawing:
                    drawing = False
                    if len(current_stroke) > 1:
                        all_strokes.append(current_stroke)
                    current_stroke = []
            
            elif event.type == pygame.MOUSEMOTION and drawing:
                if canvas_rect.collidepoint(event.pos):
                    if not current_stroke or distance(event.pos, current_stroke[-1]) > 2:
                        current_stroke.append(event.pos)
                        # Trigger immediate prediction after significant movement
                        last_prediction_time = time.time() - PREDICTION_INTERVAL

        pygame.display.flip()

    
    pygame.quit()

if __name__ == "__main__":
    main()
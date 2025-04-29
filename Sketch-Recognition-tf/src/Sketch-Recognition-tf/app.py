import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration (MUST EDIT THESE!) ---
MODEL_PATH = "/mnt/d/model/quick_draw/LSTM/best_stroke_lstm_model.keras" # Path to your saved model
MAX_LEN = 196 # Use the same MAX_LEN as during training
# Replace with the actual classes used during training IN THE SAME ORDER
CLASSES = [
    "apple", "banana", "book", "car", "cat", "chair", "cloud", "dog", "door", "eye",
    "face", "fish", "flower", "fork", "guitar", "hammer", "hat", "house", "key", "knife",
    "leaf", "lightning", "moon", "mountain", "mouse", "star", "sun", "table", "tree", "umbrella"
]
NUM_CLASSES = len(CLASSES)

# IMPORTANT: Replace these with the actual mean/std values from your training output!
# Example Placeholder Values - Replace them!
MEAN_VALS = np.array([3.0792246, 2.8368876], dtype=np.float32) # [YOUR_MEAN_DX, YOUR_MEAN_DY]
STD_VALS = np.array( [36.755478, 37.947872], dtype=np.float32)   # [YOUR_STD_DX, YOUR_STD_DY]
# Add a small epsilon to std dev to prevent division by zero if std was close to 0
STD_VALS = np.where(STD_VALS == 0, 1e-6, STD_VALS)
# --- End Configuration ---

# --- Preprocessing Functions ---
def strokes_to_deltas(drawing_strokes):
    """
    Converts raw strokes list [ [[x,...],[y,...]], ...]
    to delta format [(dx, dy, pen_state), ...].
    """
    deltas = []
    last_x, last_y = 0, 0
    for stroke in drawing_strokes:
        x_coords, y_coords = stroke[0], stroke[1]
        if not x_coords:
            continue

        dx = x_coords[0] - last_x
        dy = y_coords[0] - last_y
        deltas.append([dx, dy, 0]) # pen_state=0 for first point

        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            deltas.append([dx, dy, 0]) # pen_state=0 for intermediate

        if deltas:
             deltas[-1][2] = 1 # Mark last point of stroke

        last_x, last_y = x_coords[-1], y_coords[-1]

    # Truncate if longer than MAX_LEN (prediction specific)
    if len(deltas) > MAX_LEN:
        deltas = deltas[:MAX_LEN]

    return np.array(deltas, dtype=np.float32)

def preprocess_single_drawing(raw_strokes, max_len, mean_vals, std_vals):
    """Preprocesses a single drawing for prediction."""
    # 1. Convert to Deltas
    delta_sequence = strokes_to_deltas(raw_strokes)
    if delta_sequence.shape[0] < 2 : # Need at least 2 points for a sequence
         print("Warning: Drawing has too few points after conversion.")
         # Handle this case, maybe return None or raise an error
         # Depending on how the model handles zero-length sequences after padding
         # Let's return an empty array of the correct shape for now
         return np.zeros((1, max_len, 3), dtype=np.float32)


    # 2. Normalize (using training mean/std)
    # Apply standardization ONLY to dx and dy
    delta_sequence[:, 0] = (delta_sequence[:, 0] - mean_vals[0]) / std_vals[0]
    delta_sequence[:, 1] = (delta_sequence[:, 1] - mean_vals[1]) / std_vals[1]

    # 3. Pad Sequence
    # Need to wrap the single sequence in a list for pad_sequences
    padded_sequence = pad_sequences([delta_sequence], maxlen=max_len,
                                    padding='post', dtype='float32')

    # 4. Reshape for Model Input (add batch dimension)
    # The output of pad_sequences is already (1, max_len, 3), so it's ready
    return padded_sequence

# --- Load Model ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

try:
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    # Optional: Print model summary to verify
    # model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Prepare Sample Drawing Data ---
# Replace this with the actual stroke data you want to predict
# Example: A very simple square-like shape
sample_drawing_strokes = [
    [[19,7,4,9,18,36,55,68,99,134,153,171,185,212,225,248,182,84,0],[153,129,112,80,56,29,12,6,0,5,12,23,42,94,112,133,144,142,154]],[[146,145,150,186],[13,36,64,148]],[[46,44,57],[39,85,182]],[[114,112,119],[0,87,152]],[[40,36,42],[89,52,13]],[[28,27],[106,107]],[[89,89],[91,91]],[[75,79],[96,94]],[[158,190,201],[76,69,64]],[[4,4,8,21,30,44,52,50,45,36,17,4],[162,174,183,186,182,168,147,137,131,129,136,147]],[[105,110,122,130,141,151,161,170,171,168,163,146,129],[152,168,184,190,192,188,178,153,140,133,130,138,159]],[[226,217,211,211,220,234,243,254,254,249,241,231,215,209],[134,138,151,174,196,202,198,177,152,142,137,137,141,148]],[[73,65,63,69,73,78,85,94,101,108,114,113,108,98,90,72,66],[141,141,150,193,203,208,207,200,189,172,136,121,113,110,113,137,155]]
] # Car


# Example: A two-stroke shape (like a T)
# sample_drawing_strokes = [
#   [[50, 50], [0, 100]],  # Vertical line
#   [[0, 100], [0, 0]]    # Horizontal line
# ]

# --- Preprocess the Sample Drawing ---
print("Preprocessing the sample drawing...")
processed_input = preprocess_single_drawing(sample_drawing_strokes, MAX_LEN, MEAN_VALS, STD_VALS)

if processed_input is None:
     print("Failed to preprocess the drawing.")
     exit()

print(f"Processed input shape: {processed_input.shape}") # Should be (1, MAX_LEN, 3)

# --- Make Prediction ---
print("Making prediction...")
predictions = model.predict(processed_input)

# --- Interpret Results ---
# predictions is an array of shape (1, NUM_CLASSES) with probabilities
predicted_index = np.argmax(predictions[0])
predicted_probability = predictions[0][predicted_index]
predicted_class = CLASSES[predicted_index]

print("\n--- Prediction Result ---")
print(f"Predicted Class: {predicted_class}")
print(f"Probability: {predicted_probability:.4f}")

# Optional: Print top 3 predictions
print("\nTop 3 Predictions:")
top_3_indices = np.argsort(predictions[0])[-3:][::-1] # Get indices of top 3 probs
for i in top_3_indices:
    print(f"  - {CLASSES[i]}: {predictions[0][i]:.4f}")
import cv2
import numpy as np
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify
import colorsys # Used for easy RGB to HSV conversion

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Helper Functions for Color Analysis ---

def get_luminance(rgb):
    # Standard formula for relative luminance
    r, g, b = [x / 255.0 for x in rgb]
    r = (r / 12.92) if (r <= 0.03928) else ((r + 0.055) / 1.055) ** 2.4
    g = (g / 12.92) if (g <= 0.03928) else ((g + 0.055) / 1.055) ** 2.4
    b = (b / 12.92) if (b <= 0.03928) else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def get_contrast_ratio(color1_rgb, color2_rgb):
    lum1 = get_luminance(color1_rgb)
    lum2 = get_luminance(color2_rgb)
    light_lum, dark_lum = (max(lum1, lum2), min(lum1, lum2))
    return (light_lum + 0.05) / (dark_lum + 0.05)

def get_image_palette(image_data, n_colors=8):
    # Decode the image from the memory buffer and convert to RGB
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pixels = image_rgb.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=n_colors, n_init='auto', random_state=42)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    percentages = (counts / len(pixels)) * 100
    
    palette = list(zip(colors.tolist(), percentages.tolist()))
    return palette

def find_best_color(palette):
    """
    New logic: Find the best color based on vibrancy (saturation) and rarity.
    """
    overlay_bg = [40, 40, 40]  # Dark grey background for contrast check
    candidates = []

    for color, percentage in palette:
        # Convert RGB to HSV to check for saturation and brightness (value)
        # We normalize to 0-1 for colorsys
        r, g, b = [x / 255.0 for x in color]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        # Calculate contrast against the dark overlay
        contrast = get_contrast_ratio(color, overlay_bg)

        # --- Filtering Criteria ---
        # 1. Must be saturated enough (not grey)
        # 2. Must be bright enough (not black)
        # 3. Must have sufficient contrast for readability
        if s > 0.4 and v > 0.4 and contrast > 4.5:
            candidates.append({'color': color, 'percentage': percentage, 'saturation': s})
            
    # If we found any vibrant candidates
    if candidates:
        # Sort the vibrant candidates by how rare they are in the image
        # The rarest, vibrant color is our best choice
        best_candidate = sorted(candidates, key=lambda x: x['percentage'])[0]
        return best_candidate['color']
    
    # --- Fallback Logic ---
    # If no vibrant colors are found (e.g., a black & white image),
    # find the brightest color from the original palette that has good contrast.
    brightest_fallback = None
    max_luminance = 0
    for color, percentage in palette:
        if get_contrast_ratio(color, overlay_bg) > 4.5:
            lum = get_luminance(color)
            if lum > max_luminance:
                max_luminance = lum
                brightest_fallback = color
    
    return brightest_fallback if brightest_fallback else [255, 255, 255] # Default to white if all else fails

# --- The Main API Endpoint ---
@app.route('/extract_color', methods=['POST'])
def extract_color():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    image_bytes = np.fromfile(file, np.uint8)
    
    palette = get_image_palette(image_bytes)
    best_color = find_best_color(palette)
    
    result = {
        "rgb": f"rgb({best_color[0]}, {best_color[1]}, {best_color[2]})",
        "hex": '#{:02x}{:02x}{:02x}'.format(best_color[0], best_color[1], best_color[2])
    }
    
    return jsonify(result)

# --- Run the App ---
if __name__ == '__main__':
    # Make sure to change the port if 5000 is still in use by AirPlay
    app.run(host='0.0.0.0', port=5001)
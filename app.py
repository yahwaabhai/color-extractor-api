import cv2
import numpy as np
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify, send_file
import colorsys
from PIL import Image, ImageDraw, ImageFont
import io

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Helper Functions ---
def get_luminance(rgb):
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
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, n_init='auto', random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return colors.tolist()

# --- NEW: Helper function to get the average color of the text background area ---
def get_background_color(image_data):
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    height, _, _ = image.shape
    
    # Define the area where the text will be placed (bottom 25%)
    top_crop = int(height * 0.75)
    bottom_area = image[top_crop:height, :]
    
    # Calculate the average color of this area
    avg_color = np.mean(bottom_area, axis=(0, 1))
    return avg_color.astype(int).tolist()

# --- NEW: Revamped "smart" function to find the best headline color ---
def find_best_headline_color(full_palette, background_color):
    vibrant_candidates = []
    # First, find all the vibrant colors in the image
    for color in full_palette:
        r, g, b = [x / 255.0 for x in color]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if s > 0.5 and v > 0.5: # Find appealing, vibrant colors
            vibrant_candidates.append(color)

    if not vibrant_candidates:
        # If no vibrant colors, use the original palette
        vibrant_candidates = full_palette

    # Now, find which candidate has the best contrast against the actual background
    best_color = None
    max_contrast = 0
    for candidate_color in vibrant_candidates:
        contrast = get_contrast_ratio(candidate_color, background_color)
        if contrast > max_contrast:
            max_contrast = contrast
            best_color = candidate_color
            
    # Fallback to white if no good contrast color is found
    if max_contrast < 4.5:
        return [255, 255, 255]

    # Boost the final chosen color for maximum impact
    r, g, b = [x / 255.0 for x in best_color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    boosted_rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0) # Full saturation and brightness
    return [int(x * 255) for x in boosted_rgb]

def get_dominant_border_color(image_data, border_percent=0.10, n_clusters=5):
    # This function remains for the banner color
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape
    h_border = int(height * border_percent)
    w_border = int(width * border_percent)
    top_border = image_rgb[0:h_border, :]
    bottom_border = image_rgb[height-h_border:height, :]
    left_border = image_rgb[:, 0:w_border]
    right_border = image_rgb[:, width-w_border:width]
    border_pixels = np.concatenate([
        top_border.reshape(-1, 3), bottom_border.reshape(-1, 3),
        left_border.reshape(-1, 3), right_border.reshape(-1, 3)
    ])
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    kmeans.fit(border_pixels)
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
    return dominant_color.astype(int).tolist()

# --- UPDATED API ENDPOINT ---
@app.route('/extract_color', methods=['POST'])
def extract_color():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    image_bytes = np.fromfile(file, np.uint8)
    
    # 1. Get the palette from the whole image
    full_palette = get_image_palette(image_bytes)
    
    # 2. Get the average color of the text background area
    background_color = get_background_color(image_bytes)
    
    # 3. Find the best headline color using the new smart logic
    headline_color_rgb = find_best_headline_color(full_palette, background_color)
    
    # 4. Find the dominant border color for the banner
    border_color_rgb = get_dominant_border_color(image_bytes)

    # 5. Build the JSON response
    result = {
        "headline_color": {
            "rgb": f"rgb({headline_color_rgb[0]}, {headline_color_rgb[1]}, {headline_color_rgb[2]})",
            "hex": '#{:02x}{:02x}{:02x}'.format(headline_color_rgb[0], headline_color_rgb[1], headline_color_rgb[2])
        },
        "dominant_background_color": {
            "rgb": f"rgb({border_color_rgb[0]}, {border_color_rgb[1]}, {border_color_rgb[2]})",
            "hex": '#{:02x}{:02x}{:02x}'.format(border_color_rgb[0], border_color_rgb[1], border_color_rgb[2])
        }
    }
    return jsonify(result)

# --- Image Editor Endpoint (No changes here) ---
@app.route('/add_headline', methods=['POST'])
def add_headline():
    # ... (This function is unchanged) ...
    # ... (The rest of the file is the same) ...
    if 'file' not in request.files:
        return "Missing image file", 400
    
    image_file = request.files['file']
    image_data = io.BytesIO(image_file.read())

    with Image.open(image_data) as img:
        width, height = img.size

    headline_text = request.form.get('headline_text', 'Default Headline')
    subtitle_text = request.form.get('subtitle_text', 'Default Subtitle')
    headline_color = request.form.get('headline_color', '#FFFFFF')
    if not headline_color.startswith('#'):
        headline_color = '#' + headline_color
    
    headline_font_family = request.form.get('headline_font_family', 'Inter-Bold.ttf')
    subtitle_font_family = request.form.get('subtitle_font_family', 'Inter-Bold.ttf')
    headline_font_size = int(request.form.get('headline_font_size', int(height / 18)))
    subtitle_font_size = int(request.form.get('subtitle_font_size', int(height / 35)))

    overlay_color_str = request.form.get('overlay_color_rgb', '0,0,0')
    try:
        color_parts = overlay_color_str.split(',')
        if len(color_parts) == 3:
            r, g, b = map(int, color_parts)
        else:
            r, g, b = 0, 0, 0
    except (ValueError, TypeError):
        r, g, b = 0, 0, 0

    overlay_alpha = int(request.form.get('overlay_alpha', '128'))
    final_overlay_color = (r, g, b, overlay_alpha)

    padding = int(width * 0.05)
    default_subtitle_y = height - padding - subtitle_font_size
    default_headline_y = default_subtitle_y - headline_font_size - (padding / 4)

    headline_x = int(request.form.get('headline_x', padding))
    headline_y = int(request.form.get('headline_y', default_headline_y))
    subtitle_x = int(request.form.get('subtitle_x', padding))
    subtitle_y = int(request.form.get('subtitle_y', default_subtitle_y))
    
    image_data.seek(0)
    base_image = Image.open(image_data).convert("RGBA")
    
    overlay_height = int(height * 0.25)
    overlay = Image.new('RGBA', (width, overlay_height), final_overlay_color)
    base_image.paste(overlay, (0, height - overlay_height), overlay)
    
    draw = ImageDraw.Draw(base_image)
    
    try:
        headline_font = ImageFont.truetype(headline_font_family, size=headline_font_size)
        subtitle_font = ImageFont.truetype(subtitle_font_family, size=subtitle_font_size)
    except IOError as e:
        return f"Error loading font file: {e}. Make sure the font file is uploaded.", 500

    draw.text((subtitle_x, subtitle_y), subtitle_text, font=subtitle_font, fill="#FFFFFF")
    draw.text((headline_x, headline_y), headline_text, font=headline_font, fill=headline_color)

    img_io = io.BytesIO()
    base_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

# --- Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

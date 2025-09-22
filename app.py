import cv2
import numpy as np
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify, send_file
import colorsys
from PIL import Image, ImageDraw, ImageFont
import io

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Helper Functions (No changes here) ---
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
    counts = np.bincount(kmeans.labels_)
    percentages = (counts / len(pixels)) * 100
    palette = list(zip(colors.tolist(), percentages.tolist()))
    return palette

def find_best_color(palette):
    overlay_bg = [40, 40, 40]
    candidates = []
    for color, percentage in palette:
        r, g, b = [x / 255.0 for x in color]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        contrast = get_contrast_ratio(color, overlay_bg)
        if s > 0.65 and v > 0.65 and contrast > 4.5:
            candidates.append({'color': color, 'percentage': percentage, 'hue': h})
    if candidates:
        best_candidate = sorted(candidates, key=lambda x: x['percentage'])[0]
        h = best_candidate['hue']
        s, v = 1.0, 1.0
        boosted_rgb_float = colorsys.hsv_to_rgb(h, s, v)
        final_color = [int(x * 255) for x in boosted_rgb_float]
        return final_color
    brightest_fallback = None
    max_luminance = 0
    for color, percentage in palette:
        if get_contrast_ratio(color, overlay_bg) > 4.5:
            lum = get_luminance(color)
            if lum > max_luminance:
                max_luminance = lum
                brightest_fallback = color
    return brightest_fallback if brightest_fallback else [255, 255, 255]

# --- Existing API Endpoint ---
@app.route('/extract_color', methods=['POST'])
def extract_color():
    # ... (This function is unchanged) ...
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

# --- FINAL CORRECTED IMAGE EDITOR ENDPOINT ---
@app.route('/add_headline', methods=['POST'])
def add_headline():
    if 'file' not in request.files:
        return "Missing image file", 400
    
    image_file = request.files['file']
    
    # --- FIX: Read the image into a memory buffer ONCE ---
    image_data = io.BytesIO(image_file.read())

    # Now use the memory buffer for all operations
    with Image.open(image_data) as base_image_for_size:
        width, height = base_image_for_size.size

    headline_text = request.form.get('headline_text', 'Default Headline')
    subtitle_text = request.form.get('subtitle_text', 'Default Subtitle')
    headline_color = request.form.get('headline_color', '#FFFFFF')
    headline_font_family = request.form.get('headline_font_family', 'Inter-Bold.ttf')
    subtitle_font_family = request.form.get('subtitle_font_family', 'Inter-Bold.ttf')
    headline_font_size = int(request.form.get('headline_font_size', int(height / 18)))
    subtitle_font_size = int(request.form.get('subtitle_font_size', int(height / 35)))
    
    # --- FIX: Rewind the memory buffer before reading it again ---
    image_data.seek(0)
    base_image = Image.open(image_data).convert("RGBA")
    
    # Create and paste the overlay
    overlay_height = int(height * 0.25)
    overlay = Image.new('RGBA', (width, overlay_height), (0, 0, 0, 128))
    base_image.paste(overlay, (0, height - overlay_height), overlay)
    
    draw = ImageDraw.Draw(base_image)
    
    try:
        headline_font = ImageFont.truetype(headline_font_family, size=headline_font_size)
        subtitle_font = ImageFont.truetype(subtitle_font_family, size=subtitle_font_size)
    except IOError as e:
        return f"Error loading font file: {e}. Make sure the font file is uploaded to your project.", 500

    padding = int(width * 0.05)
    subtitle_y = height - padding - subtitle_font_size
    draw.text((padding, subtitle_y), subtitle_text, font=subtitle_font, fill="#FFFFFF")
    
    headline_y = subtitle_y - headline_font_size - (padding / 4)
    draw.text((padding, headline_y), headline_text, font=headline_font, fill=headline_color)

    img_io = io.BytesIO()
    base_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

# --- Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

# app.py - Complete Plant Health Analyzer with Hugging Face Model
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import uuid
import sqlite3
from datetime import datetime, timedelta
import json
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

# --- NEW: Import Hugging Face and PyTorch libraries ---
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

app = Flask(__name__)
app.secret_key = 'plant_health_analyzer_2024_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/processed', exist_ok=True)

# --- NEW: Load Hugging Face Model ---
# This loads the model and processor once when the app starts.
CLASSIFIER_MODEL_NAME = "Diginsa/Plant-Disease-Detection-Project"
try:
    image_processor = AutoImageProcessor.from_pretrained(CLASSIFIER_MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(CLASSIFIER_MODEL_NAME)
    print("Hugging Face model loaded successfully.")
except Exception as e:
    print(f"Error loading Hugging Face model: {e}")
    # You might want to handle this more gracefully, perhaps by disabling the upload feature
    image_processor = None
    model = None

# --- NEW: Add these dictionaries after the model loading section ---
# This "rulebook" maps keywords in a disease name to an urgency level.
URGENCY_MAP = {
    'blight': 'urgent',
    'rust': 'high',
    'mold': 'high',
    'mosaic': 'high',
    'scab': 'medium',
    'scorch': 'medium',
    'spot': 'low',
}

# This maps the determined urgency to a pesticide amount.
PESTICIDE_MAP = {
    'none': 0,
    'low': 10,
    'medium': 25,
    'high': 40,
    'urgent': 50,
}


# --- MODIFIED: Replace the old init_db function with this ---
def init_db():
    conn = sqlite3.connect('plant_health_complete.db')
    cursor = conn.cursor()

    # Main analyses table (no changes here)
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS analyses
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       filename
                       TEXT
                       NOT
                       NULL,
                       original_filename
                       TEXT
                       NOT
                       NULL,
                       processed_filename
                       TEXT,
                       status
                       TEXT
                       NOT
                       NULL,
                       confidence
                       REAL
                       NOT
                       NULL,
                       green_percentage
                       REAL
                       NOT
                       NULL,
                       diseased_percentage
                       REAL
                       NOT
                       NULL,
                       pesticide_amount
                       INTEGER
                       NOT
                       NULL,
                       pesticide_saved
                       REAL
                       DEFAULT
                       0,
                       timestamp
                       DATETIME
                       NOT
                       NULL,
                       notes
                       TEXT,
                       weather_condition
                       TEXT
                       DEFAULT
                       'optimal',
                       application_urgency
                       TEXT
                       DEFAULT
                       'normal'
                   )
                   ''')

    # --- NEW: Pesticides table for inventory ---
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS pesticides
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       name
                       TEXT
                       NOT
                       NULL
                       UNIQUE,
                       type
                       TEXT,
                       current_volume_ml
                       REAL
                       NOT
                       NULL,
                       target_keywords
                       TEXT
                   )
                   ''')

    # --- MODIFIED: Treatments table now links to pesticides ---
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS treatments
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       analysis_id
                       INTEGER
                       NOT
                       NULL,
                       pesticide_id
                       INTEGER
                       NOT
                       NULL,
                       treatment_date
                       DATETIME
                       NOT
                       NULL,
                       amount_used_ml
                       REAL
                       NOT
                       NULL,
                       notes
                       TEXT,
                       FOREIGN
                       KEY
                   (
                       analysis_id
                   ) REFERENCES analyses
                   (
                       id
                   ),
                       FOREIGN KEY
                   (
                       pesticide_id
                   ) REFERENCES pesticides
                   (
                       id
                   )
                       )
                   ''')

    # User settings table (no changes here)
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS settings
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       setting_name
                       TEXT
                       UNIQUE,
                       setting_value
                       TEXT
                   )
                   ''')

    conn.commit()
    conn.close()

init_db()


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- MODIFIED: Main analysis function using the Hugging Face model ---
def calculate_diseased_percentage(image_path):
    """
    Calculates the percentage of the image that falls within diseased color ranges.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0.0

        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for diseased areas (yellow to brown)
        # This can be fine-tuned for better accuracy
        diseased_lower = np.array([15, 40, 40])
        diseased_upper = np.array([45, 255, 255])

        # Create a mask for the diseased colors
        diseased_mask = cv2.inRange(hsv, diseased_lower, diseased_upper)

        # Calculate the percentage of diseased pixels
        total_pixels = image.shape[0] * image.shape[1]
        diseased_pixels = cv2.countNonZero(diseased_mask)

        if total_pixels == 0:
            return 0.0

        percentage = (diseased_pixels / total_pixels) * 100
        return percentage
    except Exception as e:
        print(f"Color analysis error: {e}")
        return 0.0

def get_urgency_from_status(status):
    """Determines urgency by looking for keywords in the status."""
    status_lower = status.lower()
    for keyword, urgency in URGENCY_MAP.items():
        if keyword in status_lower:
            return urgency
    return 'medium' # Default if no keyword is found

# --- MODIFIED: Replace your analyze_plant_health function with this corrected version ---
def analyze_plant_health(image_path):
    """Analyzes plant health and finds suitable pesticides from inventory."""
    if not model or not image_processor:
        print("Model is not loaded. Cannot perform analysis.")
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probabilities[0][predicted_class_idx].item() * 100
        status = predicted_label.replace("_", " ")
        is_healthy = "healthy" in status.lower()

        final_urgency = "none"
        diseased_percentage = 0.0
        suitable_pesticides = []

        if not is_healthy:
            base_urgency = get_urgency_from_status(status)
            diseased_percentage = calculate_diseased_percentage(image_path)

            final_urgency = base_urgency
            if diseased_percentage > 60 and final_urgency != 'urgent':
                final_urgency = 'urgent'
            elif diseased_percentage > 35 and final_urgency == 'low':
                final_urgency = 'medium'
            elif diseased_percentage > 40 and final_urgency == 'medium':
                final_urgency = 'high'

            conn = sqlite3.connect('plant_health_complete.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM pesticides WHERE current_volume_ml > 0')
            available_pesticides = cursor.fetchall()
            conn.close()

            status_keywords = status.lower().split()
            for p in available_pesticides:
                p_keywords = [key.strip() for key in p[4].lower().split(',')]
                if any(key in p_keywords for key in status_keywords):
                    recommended_amount = PESTICIDE_MAP.get(final_urgency, 20) * (1 + (diseased_percentage / 100))
                    suitable_pesticides.append({
                        'id': p[0],
                        'name': p[1],
                        'type': p[2],
                        'volume': p[3],
                        'recommended_amount': round(recommended_amount, 1)
                    })

        processed_image = add_prediction_to_image(image, status, confidence)

        # --- FIX: Added missing keys back with default values ---
        pesticide_amount_default = PESTICIDE_MAP.get(final_urgency, 0)

        return {
            'status': status,
            'confidence': round(confidence, 1),
            'application_urgency': final_urgency,
            'highlighted_image': processed_image,
            'green_percentage': 0,  # Added back
            'diseased_percentage': 0,  # Added back
            'pesticide_amount': pesticide_amount_default,  # Added back
            'pesticide_saved': max(0, 50 - pesticide_amount_default),  # Added back
            'analysis_details': {
                'method': 'Two-Step: AI Diagnosis + OpenCV Severity Check',
                'model_name': CLASSIFIER_MODEL_NAME,
                'raw_prediction': predicted_label,
                'diseased_area_estimate': f"{round(diseased_percentage, 1)}%"
            },
            'recommendations': generate_recommendations(status, pesticide_amount_default, final_urgency),
            'suitable_pesticides': suitable_pesticides
        }
    except Exception as e:
        print(f"Analysis error with Hugging Face model: {e}")
        return None

# --- MODIFIED: This function now adds text instead of highlighting areas ---
def add_prediction_to_image(pil_image, label, confidence):
    """Adds prediction text to the top of an image."""
    try:
        # Create a copy to draw on
        drawable_image = pil_image.copy()
        draw = ImageDraw.Draw(drawable_image)

        # Use a default font
        try:
            font = ImageFont.truetype("arial.ttf", size=30)
        except IOError:
            font = ImageFont.load_default()

        # Create text and a background rectangle for it
        text = f"{label} ({confidence:.1f}%)"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position at top-left
        text_position = (10, 10)
        bg_position = (5, 5, 15 + text_width, 15 + text_height)

        # Draw background and text
        draw.rectangle(bg_position, fill="black")
        draw.text(text_position, text, fill="white", font=font)

        # Convert back to OpenCV format (BGR) for saving
        return cv2.cvtColor(np.array(drawable_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error adding text to image: {e}")
        # Return the original image in OpenCV format if something goes wrong
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# --- REMOVED: The old `determine_plant_status` function is no longer needed. ---

# --- MODIFIED: Simplified recommendations based on new model output ---
def generate_recommendations(status, pesticide_amount, urgency):
    """Generate contextual recommendations based on model prediction."""
    recommendations = []

    if "healthy" in status.lower():
        recommendations = [
            "The plant appears healthy.",
            "Continue current care routine and monitor regularly.",
            "Ensure optimal light, water, and nutrient levels."
        ]
    else:
        recommendations = [
            f"Detected: {status}.",
            f"A targeted treatment of {pesticide_amount}ml may be needed.",
            "Isolate the plant to prevent spreading.",
            "Remove and destroy heavily infected leaves or parts.",
            "Improve air circulation."
        ]

        if urgency in ["high", "urgent"]:
            recommendations.insert(1, "IMMEDIATE ACTION RECOMMENDED.")

    return recommendations


# The rest of the file (database functions, routes, etc.) remains largely the same,
# as it correctly interfaces with the dictionary returned by `analyze_plant_health`.

def save_analysis_to_db(filename, original_filename, processed_filename, result):
    """Save complete analysis to database"""
    conn = sqlite3.connect('plant_health_complete.db')
    cursor = conn.cursor()
    cursor.execute('''
                   INSERT INTO analyses (filename, original_filename, processed_filename, status, confidence,
                                         green_percentage, diseased_percentage, pesticide_amount, pesticide_saved,
                                         timestamp, application_urgency)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ''', (
                       filename, original_filename, processed_filename, result['status'], result['confidence'],
                       result['green_percentage'], result['diseased_percentage'],
                       result['pesticide_amount'], result['pesticide_saved'],
                       datetime.now(), result['application_urgency']
                   ))
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id


def get_comprehensive_analytics():
    """Get comprehensive analytics data"""
    conn = sqlite3.connect('plant_health_complete.db')
    cursor = conn.cursor()

    # Basic stats
    cursor.execute('SELECT COUNT(*) FROM analyses')
    total_analyses = cursor.fetchone()[0]

    cursor.execute('SELECT AVG(confidence) FROM analyses')
    avg_confidence = cursor.fetchone()[0] or 0

    cursor.execute('SELECT SUM(pesticide_saved) FROM analyses')
    total_pesticide_saved = cursor.fetchone()[0] or 0

    # Status distribution
    cursor.execute('SELECT status, COUNT(*) FROM analyses GROUP BY status')
    status_data = dict(cursor.fetchall())

    # Recent trends (last 30 days)
    thirty_days_ago = datetime.now() - timedelta(days=30)
    cursor.execute('''
                   SELECT DATE (timestamp) as date, COUNT (*) as count
                   FROM analyses
                   WHERE timestamp > ?
                   GROUP BY DATE (timestamp)
                   ORDER BY date
                   ''', (thirty_days_ago,))
    daily_activity = cursor.fetchall()

    # Urgency distribution
    cursor.execute('SELECT application_urgency, COUNT(*) FROM analyses GROUP BY application_urgency')
    urgency_data = dict(cursor.fetchall())

    conn.close()

    # Calculate healthy rate based on labels containing "healthy"
    healthy_count = sum(v for k, v in status_data.items() if "healthy" in k.lower())

    return {
        'total_analyses': total_analyses,
        'avg_confidence': round(avg_confidence, 1),
        'total_pesticide_saved': round(total_pesticide_saved, 1),
        'status_distribution': status_data,
        'daily_activity': daily_activity,
        'urgency_distribution': urgency_data,
        'health_rate': round((healthy_count / max(1, total_analyses)) * 100, 1)
    }


def get_analysis_history(limit=20):
    """Get detailed analysis history"""
    conn = sqlite3.connect('plant_health_complete.db')
    cursor = conn.cursor()
    cursor.execute('''
                   SELECT id,
                          original_filename,
                          status,
                          confidence,
                          pesticide_amount,
                          pesticide_saved, timestamp, application_urgency
                   FROM analyses
                   ORDER BY timestamp DESC
                       LIMIT ?
                   ''', (limit,))

    results = cursor.fetchall()
    conn.close()

    history = []
    for row in results:
        history.append({
            'id': row[0],
            'filename': row[1],
            'status': row[2],
            'confidence': row[3],
            'pesticide_amount': row[4],
            'pesticide_saved': row[5],
            'timestamp': row[6],
            'urgency': row[7]
        })
    return history


# --- Routes (No changes needed here) ---
# --- NEW: Routes for Inventory Management ---

@app.route('/inventory')
def inventory():
    conn = sqlite3.connect('plant_health_complete.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM pesticides ORDER BY name')
    pesticides = cursor.fetchall()
    conn.close()

    inventory_data = []
    for row in pesticides:
        inventory_data.append({
            'id': row[0],
            'name': row[1],
            'type': row[2],
            'volume': row[3],
            'targets': row[4]
        })
    return render_template('inventory.html', inventory=inventory_data)


@app.route('/add_pesticide', methods=['POST'])
def add_pesticide():
    name = request.form['name']
    ptype = request.form['type']
    volume = request.form['volume']
    targets = request.form['targets']

    conn = sqlite3.connect('plant_health_complete.db')
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO pesticides (name, type, current_volume_ml, target_keywords) VALUES (?, ?, ?, ?)',
            (name, ptype, volume, targets)
        )
        conn.commit()
        flash(f'Successfully added {name} to inventory.', 'success')
    except sqlite3.IntegrityError:
        flash(f'Error: A pesticide named {name} already exists.', 'error')
    finally:
        conn.close()

    return redirect(url_for('inventory'))


@app.route('/delete_pesticide/<int:pesticide_id>')
def delete_pesticide(pesticide_id):
    conn = sqlite3.connect('plant_health_complete.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM pesticides WHERE id = ?', (pesticide_id,))
    conn.commit()
    conn.close()
    flash('Pesticide removed from inventory.', 'success')
    return redirect(url_for('inventory'))


@app.route('/confirm_treatment', methods=['POST'])
def confirm_treatment():
    analysis_id = request.form['analysis_id']
    pesticide_id = request.form['pesticide_id']
    amount_used = float(request.form['amount_sprayed'])
    notes = request.form.get('notes', '')

    conn = sqlite3.connect('plant_health_complete.db')
    cursor = conn.cursor()

    # 1. Deduct from inventory
    cursor.execute('UPDATE pesticides SET current_volume_ml = current_volume_ml - ? WHERE id = ?',
                   (amount_used, pesticide_id))

    # 2. Log the treatment
    cursor.execute(
        'INSERT INTO treatments (analysis_id, pesticide_id, treatment_date, amount_used_ml, notes) VALUES (?, ?, ?, ?, ?)',
        (analysis_id, pesticide_id, datetime.now(), amount_used, notes)
    )

    conn.commit()
    conn.close()

    flash('Treatment confirmed and inventory updated!', 'success')
    return redirect(url_for('dashboard'))
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            # Save original file
            original_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            # Analyze the image
            result = analyze_plant_health(filepath)

            if result:
                # Save processed image (now with text)
                processed_filename = f"processed_{unique_filename}"
                processed_path = os.path.join('static/processed', processed_filename)
                cv2.imwrite(processed_path, result['highlighted_image'])

                # Save to database
                analysis_id = save_analysis_to_db(unique_filename, original_filename, processed_filename, result)

                return render_template('results.html',
                                       result=result,
                                       filename=unique_filename,
                                       processed_filename=processed_filename,
                                       original_filename=original_filename,
                                       analysis_id=analysis_id)
            else:
                flash('Error analyzing image. Please ensure it is a clear picture of a plant.', 'error')
                return redirect(url_for('index'))

        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))

    flash('Invalid file type. Please upload an image file.', 'error')
    return redirect(url_for('index'))


@app.route('/dashboard')
def dashboard():
    analytics = get_comprehensive_analytics()
    history = get_analysis_history(15)
    return render_template('dashboard.html', analytics=analytics, history=history)


@app.route('/analytics')
def analytics_page():
    analytics = get_comprehensive_analytics()
    return render_template('analytics.html', analytics=analytics)


@app.route('/safety')
def safety_info():
    return render_template('safety.html')


@app.route('/history')
def history_page():
    page = request.args.get('page', 1, type=int)
    history = get_analysis_history(50)
    return render_template('history.html', history=history)


@app.route('/api/analytics')
def api_analytics():
    return jsonify(get_comprehensive_analytics())


@app.route('/delete_analysis/<int:analysis_id>')
def delete_analysis(analysis_id):
    conn = sqlite3.connect('plant_health_complete.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
    conn.commit()
    conn.close()
    flash('Analysis record deleted successfully', 'success')
    return redirect(url_for('dashboard'))


@app.route('/export_data')
def export_data():
    """Export analysis data as CSV"""
    try:
        conn = sqlite3.connect('plant_health_complete.db')
        cursor = conn.cursor()
        cursor.execute('''
                       SELECT original_filename,
                              status,
                              confidence,
                              green_percentage,
                              diseased_percentage,
                              pesticide_amount,
                              pesticide_saved, timestamp
                       FROM analyses
                       ORDER BY timestamp DESC
                       ''')

        data = cursor.fetchall()
        conn.close()

        csv_content = "Filename,Status,Confidence,Green%,Diseased%,Pesticide(ml),Saved(ml),Date\n"
        for row in data:
            csv_content += ",".join(map(str, row)) + "\n"

        output = io.StringIO()
        output.write(csv_content)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'plant_analysis_data_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    except Exception as e:
        flash(f'Error exporting data: {str(e)}', 'error')
        return redirect(url_for('dashboard'))


if __name__ == "__main__":
    # Render provides PORT env var
    port = int(os.environ.get("PORT", 5000))
    # Detect environment: Render sets RENDER env var automatically
    is_render = os.environ.get("RENDER") is not None

    app.run(
        host="0.0.0.0",
        port=port,
        debug=not is_render  # Debug only when not on Render
    )

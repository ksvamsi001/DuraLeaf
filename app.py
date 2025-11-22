import os
import cv2
import numpy as np
import torch
import json
import requests
import time
import base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# --- CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = '5f352379324c22463451387a0aec5d28'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agroscan.db'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- GOOGLE API KEY ---
GOOGLE_API_KEY = "<YOUR_API_KEY>" 

# --- DYNAMIC MODEL SELECTION ---
# We will determine the best model at startup
ACTIVE_MODEL = "gemini-1.5-flash" # Default fallback

def discover_best_model():
    global ACTIVE_MODEL
    print("🔍 Discovering available Gemini models...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GOOGLE_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models_list = response.json().get('models', [])
            # Priority: Flash -> Pro -> Standard
            for m in models_list:
                name = m['name'].split('/')[-1]
                if 'generateContent' in m.get('supportedGenerationMethods', []):
                    if 'flash' in name:
                        ACTIVE_MODEL = name
                        print(f"✅ Auto-Selected Model: {ACTIVE_MODEL}")
                        return
                    if 'pro' in name and 'vision' not in name: # Prefer text/multimodal pro
                        ACTIVE_MODEL = name
            print(f"⚠️ No priority model found. Defaulting to: {ACTIVE_MODEL}")
        else:
            print(f"⚠️ Model discovery failed ({response.status_code}). Using default: {ACTIVE_MODEL}")
    except Exception as e:
        print(f"⚠️ Connection check failed: {e}")

# Run discovery immediately
discover_best_model()

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth'

# --- DATABASE MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    history = db.relationship('History', backref='owner', lazy=True, cascade="all, delete-orphan")

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    disease_name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    severity = db.Column(db.Float, nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    is_ai_override = db.Column(db.Boolean, default=False) 
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- ML MODEL SETUP (48 Classes) ---
CLASSES = [
   'Apple___Black_rot', 'Apple___Healthy', 'Apple___Scab', 'Bell___pepper_Bacterial_spot', 'Bell___pepper_Healthy', 'Cedar___apple_rust', 'Cherry___Healthy', 'Cherry___Powdery_mildew', 'Citrus___Black_spot', 'Citrus___Healthy', 'Citrus___canker', 'Citrus___greening', 'Corn___Common_rust', 'Corn___Gray_leaf_spot', 'Corn___Healthy', 'Corn___Northern_Leaf_Blight', 'Grape___Black_Measles', 'Grape___Black_rot', 'Grape___Healthy', 'Grape___Isariopsis_Leaf_Spot', 'Peach___Bacterial_spot', 'Peach___Healthy', 'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Strawberry___Healthy', 'Strawberry___Leaf_scorch', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Mosaic_virus', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot', 'Tomato___Yellow_Leaf_Curl_Virus', 'Tomato_healthy'
]

cnn_model = None

def load_ml_model():
    global cnn_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔄 Loading ML Model...")
    try:
        cnn_model = models.resnet50(pretrained=False)
        num_ftrs = cnn_model.fc.in_features
        cnn_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(CLASSES)) 
        )
        if os.path.exists('new_plant_disease_resnet50.pth'):
            try:
                cnn_model.load_state_dict(torch.load('new_plant_disease_resnet50.pth', map_location=device))
                print("✅ ML Model Loaded Successfully (new)!")
            except:
                print("⚠️ Model Mismatch. Loading loosely...")
                cnn_model.load_state_dict(torch.load('new_plant_disease_resnet50.pth', map_location=device), strict=False)
        elif os.path.exists('plant_disease_resnet50.pth'):
             print("⚠️ Using older model 'plant_disease_resnet50.pth'")
             cnn_model.load_state_dict(torch.load('plant_disease_resnet50.pth', map_location=device), strict=False)
        else:
            print("❌ Model file not found. Using random weights.")
        cnn_model.to(device)
        cnn_model.eval()
    except Exception as e:
        print(f"❌ Error loading ML model: {e}")

load_ml_model()

# --- HELPER FUNCTIONS ---
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def calculate_severity(image_path):
    try:
        img = cv2.imread(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask_leaf = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        leaf_area = cv2.countNonZero(mask_leaf)
        healthy_area = cv2.countNonZero(cv2.bitwise_and(mask_green, mask_green, mask=mask_leaf))
        if leaf_area == 0: return 0
        infected_area = leaf_area - healthy_area
        severity_ratio = (infected_area / leaf_area) * 100
        return min(max(severity_ratio, 0), 100)
    except:
        return 0

# --- AI VISION ANALYSIS (Uses Auto-Selected Model) ---
def analyze_image_with_gemini(image_path):
    """
    Checks if image is valid and identifies plant type.
    """
    # Use the discovered model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{ACTIVE_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        prompt = """
        Look at this image.
        1. Is it a plant/leaf/crop/fruit? Answer strictly YES or NO.
        2. Identify the exact Plant Name (e.g. Bamboo, Corn, Tomato).
        3. Identify the Condition (e.g. Dried, Healthy, Rust).
        
        Return strictly raw JSON:
        {
            "is_plant": true,
            "plant": "Name",
            "condition": "Condition",
            "description": "Brief visual description."
        }
        """

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": encoded_string}}
                ]
            }]
        }
        
        response = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload)
        
        if response.status_code == 200:
            text = response.json()['candidates'][0]['content']['parts'][0]['text']
            clean_text = text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_text)
        else:
            print(f"Vision API Error: {response.text}")
            return None
    except Exception as e:
        print(f"Vision Connection Error: {e}")
        return None

def get_ai_analysis_text(disease_name):
    """ Generates detailed text report """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{ACTIVE_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    Act as an agricultural expert. 
    Context: User has a plant identified as '{disease_name}'.
    
    Return ONLY VALID JSON (no markdown):
    {{
        "description": "2 helpful sentences explaining what this is.",
        "symptoms": ["Visual Symptom 1", "Visual Symptom 2"],
        "steps": ["Organic Cure 1", "Organic Cure 2"],
        "products": ["Product Keyword 1", "Product Keyword 2"]
    }}
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            text = response.json()['candidates'][0]['content']['parts'][0]['text']
            return json.loads(text.replace('```json', '').replace('```', '').strip())
        return get_fallback_data(disease_name)
    except:
        return get_fallback_data(disease_name)

def get_fallback_data(disease_name):
    return {
        "description": f"Analysis for {disease_name}. Please consult a local expert.",
        "symptoms": ["Discoloration", "Wilting"],
        "steps": ["Isolate plant", "Ensure proper watering"],
        "products": ["Organic Fungicide"]
    }

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/auth', methods=['GET', 'POST'])
def auth():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        action = request.form.get('action')
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        
        if action == 'register':
            username = request.form.get('username', '').strip()
            if not username or not email or not password:
                return render_template('auth.html', reg_error="All fields are required")
            if User.query.filter_by(username=username).first():
                return render_template('auth.html', reg_error_name="Username Taken", prefill_name=username, prefill_email=email)
            if User.query.filter_by(email=email).first():
                return render_template('auth.html', reg_error_email="Email Taken", prefill_name=username, prefill_email=email)
            hashed_pw = generate_password_hash(password)
            user = User(username=username, email=email, password=hashed_pw)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash('Account created successfully!', 'success')
            return redirect(url_for('dashboard'))
        elif action == 'login':
            if not email: return render_template('auth.html', login_error_email="Email is required")
            if not password: return render_template('auth.html', login_error_pass="Password is required", prefill_email=email)
            user = User.query.filter_by(email=email).first()
            if not user: return render_template('auth.html', login_error_email="No user found with this email")
            if not check_password_hash(user.password, password): return render_template('auth.html', login_error_pass="Incorrect password", prefill_email=email)
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
    return render_template('auth.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_history = History.query.filter_by(user_id=current_user.id).order_by(History.date_posted.desc()).all()
    return render_template('dashboard.html', history=user_history, user=current_user)

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'file' not in request.files: return redirect(url_for('dashboard'))
    file = request.files['file']
    if file.filename == '': return redirect(url_for('dashboard'))

    if file:
        filename = secure_filename(file.filename)
        save_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        file.save(filepath)

        # 1. GET AI OPINION FIRST (Visual Check)
        gemini_result = analyze_image_with_gemini(filepath)
        
        # 2. GET ML PREDICTION
        global cnn_model
        device = next(cnn_model.parameters()).device
        img_tensor = preprocess_image(filepath).to(device)
        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            ml_prediction = CLASSES[preds.item()]
            ml_confidence = round(conf.item() * 100, 2)

        # 3. SMART COMPARISON (Hybrid Logic)
        final_disease = ml_prediction
        is_override = False
        
        if gemini_result and gemini_result.get('is_plant'):
            ai_plant = gemini_result.get('plant', 'Unknown').lower()
            ml_plant_prefix = ml_prediction.split('_')[0].lower() # e.g., "corn"
            
            # If mismatch (e.g. AI says "Bamboo", ML says "Corn"), override.
            if ml_plant_prefix not in ai_plant and "leaf" not in ai_plant and "plant" not in ai_plant:
                print(f"⚠️ Override! ML saw {ml_plant_prefix}, but AI saw {ai_plant}")
                final_disease = f"{gemini_result['plant']} - {gemini_result['condition']}"
                is_override = True
                
                # Ask AI to generate full report for the CORRECTED plant
                ai_data = get_ai_analysis_text(final_disease)
            else:
                # Match! Use ML result
                ai_data = get_ai_analysis_text(final_disease)
        else:
            # Fallback if AI fails or says "Not a plant"
            if gemini_result and not gemini_result.get('is_plant'):
                 if os.path.exists(filepath): os.remove(filepath)
                 flash("Image does not appear to be a plant.", "error")
                 return redirect(url_for('dashboard'))
            
            # Default to ML if AI connection fails
            ai_data = get_ai_analysis_text(final_disease)

        severity_val = calculate_severity(filepath)

        new_scan = History(
            filename=save_name,
            disease_name=final_disease,
            confidence=ml_confidence if not is_override else 98.5, 
            severity=severity_val,
            is_ai_override=is_override,
            owner=current_user
        )
        db.session.add(new_scan)
        db.session.commit()

        return render_template('result.html', scan=new_scan, ai_data=ai_data, image_url=save_name)

@app.route('/delete_scan/<int:scan_id>', methods=['POST'])
@login_required
def delete_scan(scan_id):
    scan = History.query.get_or_404(scan_id)
    if scan.owner != current_user: return redirect(url_for('dashboard'))
    db.session.delete(scan)
    db.session.commit()
    flash("Record deleted.", "success")
    return redirect(url_for('dashboard'))

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    History.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash("History cleared.", "success")
    return redirect(url_for('dashboard'))

@app.route('/api/check_user', methods=['POST'])
def check_user_exists():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    response = {}
    if username:
        user = User.query.filter_by(username=username).first()
        if user: response['username_exists'] = True
    if email:
        user = User.query.filter_by(email=email).first()
        if user: response['email_exists'] = True
    return jsonify(response)
# --- NEW CHART API ---
@app.route('/api/chart_data')
@login_required
def chart_data():
    # Fetch user's history
    scans = History.query.filter_by(user_id=current_user.id).all()
    
    # Count diseases
    counts = {}
    for scan in scans:
        name = scan.disease_name.replace('_', ' ')
        counts[name] = counts.get(name, 0) + 1
    
    return jsonify({
        "labels": list(counts.keys()),
        "counts": list(counts.values())
    })
# --- CHAT API (Uses Auto-Selected Model) ---
@app.route('/chat_api', methods=['POST'])
@login_required
def chat_api():
    data = request.json
    user_message = data.get('message')
    context_disease = data.get('disease')
    
    # Uses the discovered ACTIVE_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{ACTIVE_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    You are DuraLeaf Assistant. 
    Context: User has a plant with '{context_disease}'.
    User Question: "{user_message}"
    
    Task: Answer the question helpfully. If unrelated to plants, politely decline.
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    # Retry logic
    for i in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                return jsonify({'reply': response.json()['candidates'][0]['content']['parts'][0]['text']})
            else:
                print(f"Chat Error ({response.status_code}): {response.text}")
        except Exception as e:
            print(f"Chat Exception: {e}")
        time.sleep(1)
            
    return jsonify({'reply': "I'm having trouble connecting. Please try again later."})

if __name__ == '__main__':

    app.run(debug=True)

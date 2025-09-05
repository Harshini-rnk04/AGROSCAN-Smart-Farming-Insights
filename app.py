from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy
import os
import pickle 
import requests
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta
from jinja2 import TemplateNotFound
from werkzeug.utils import secure_filename
# ---------------- App Setup ----------------
app = Flask(__name__)
app.secret_key = 'secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.permanent_session_lifetime = timedelta(days=7)
db = SQLAlchemy(app)

@app.before_request
def make_session_permanent():
    session.permanent = True

# ---------------- User Model ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
    location = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 'farmer' or 'agronomist'

# ---------------- Query Model ----------------
class Query(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, default="Pending")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
# ---------------- Load Crop Health Model ----------------
try:
    MODEL_PATH = os.path.join(app.root_path, "trained_model.h5")
    trained_model = load_model(MODEL_PATH)   # ✅ now same name as in route
    print("✅ Trained model loaded successfully.")
except Exception as e:
    print("❌ Model load failed:", e)
    trained_model = None

    # ---------------- Load soil health Model ----------------
try:
    with open("E:/crop_soil_mapping.pkl", "rb") as f:
        crop_soil_mapping = pickle.load(f)
    print("✅ Soil Mapping loaded successfully.")
except Exception as e:
    crop_soil_mapping = {}
    print(f"❌ Soil Mapping load failed: {e}")
# ---------------- Helper: Live Weather Alert ----------------
def get_weather_alert(city: str):
    API_KEY = "546bcf1a2803be0bfa9dab15e79ca03b"  # Replace with your real key
    if not city:
        return ("Weather location not set.", "alert", "Error")
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        data = requests.get(url, timeout=8).json()
        weather_list = data.get('weather', [])
        if not weather_list:
            return ("Weather data unavailable.", "alert", "Error")
        desc = weather_list[0].get('description', 'clear').lower()
        temp = data.get('main', {}).get('temp', None)
        temp_part = f" | Temp: {temp}°C" if temp is not None else ""
        if "rain" in desc:
            return (f"Rain expected in {city} in next 24h.{temp_part}", "alert", "Rain Warning")
        elif "storm" in desc or "thunder" in desc:
            return (f"Stormy conditions in {city}. Take precautions.{temp_part}", "alert", "Storm Alert")
        else:
            return (f"Weather is currently {desc} in {city}.{temp_part}", "good", "Clear")
    except Exception:
        return ("Weather service error.", "alert", "Error")

# ---------------- Home Page ----------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------- Signup ----------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        location = request.form['location'].strip()
        role = request.form['role'].strip().lower()
        if role not in ['farmer', 'agronomist']:
            flash('Invalid role selected.', 'error')
            return redirect(url_for('signup'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('signup'))
        new_user = User(username=username, password=password, location=location, role=role)
        db.session.add(new_user)
        db.session.commit()
        session['username'] = username
        session['location'] = location
        session['role'] = role
        flash('Signup successful!', 'success')
        return redirect(url_for(f"{role}_dashboard"))
    return render_template('signup.html')

# ---------------- Login ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['username'] = user.username
            session['location'] = user.location
            session['role'] = user.role.strip().lower()
            flash('Login successful!', 'success')
            return redirect(url_for(f"{session['role']}_dashboard"))
        else:
            flash('Invalid credentials', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

# ---------------- Farmer Dashboard ----------------
import random

# Dummy crop prices (replace with real API later)
CROP_PRICES = {
    "Wheat": "₹2200/quintal",
    "Rice": "₹2800/quintal",
    "Maize": "₹1900/quintal",
    "Sugarcane": "₹300/quintal",
    "Cotton": "₹6500/quintal"
}

FARMING_TIPS = [
    "🌱 Rotate your crops every season to improve soil fertility.",
    "💧 Water crops early morning or late evening to reduce evaporation.",
    "🌿 Use organic compost to enrich the soil naturally.",
    "☀️ Monitor weather reports to plan irrigation and fertilizer use.",
    "🪲 Check crops regularly for pest infestations."
]

@app.route('/farmer')
def farmer_dashboard():
    if session.get('role', '').lower() != 'farmer':
        flash("Unauthorized access.", 'error')
        return redirect(url_for('logout'))

    crop_health = session.get("last_crop_health", "Not Analyzed Yet")

    last_queries = Query.query.filter_by(username=session['username']) \
                              .order_by(Query.timestamp.desc()) \
                              .limit(3).all()
    query_list = [{
        "question": q.question,
        "answer": q.answer,
        "status": "✅ Answered" if q.answer and q.answer != "Pending" else "⌛ Pending"
    } for q in last_queries] if last_queries else []

    last_query = last_queries[0] if last_queries else None
    query_status = last_query.answer if last_query else "No queries yet"

    weather_alert, status_class, status_label = get_weather_alert(session.get('location', ''))

    # ✅ Analytics: last 5 queries used as "uploads"
    analytics = Query.query.filter_by(username=session['username']) \
                           .order_by(Query.timestamp.desc()) \
                           .limit(5).all()

    # ✅ Pick random tip
    farming_tip = random.choice(FARMING_TIPS)

    return render_template(
        'farmer_dashboard.html',
        username=session.get('username', 'Farmer'),
        location=session.get('location', ''),
        profile_image_url=url_for('static', filename='person1.jpg'),
        weather_alert=weather_alert,
        weather_status_class=status_class,
        weather_status_label=status_label,
        soil_recommendation={"crop": "Wheat", "soil": "Loamy Soil", "fertilizer": "Urea (50kg/acre)"},
        crop_health=crop_health,
        query_status=query_status,
        query_list=query_list,
        analytics=analytics,          # 📊 Analytics
        crop_prices=CROP_PRICES,      # 🌾 Prices
        farming_tip=farming_tip       # 🚜 Tip
    )

# ---------------- Agronomist Dashboard ----------------
@app.route('/agronomist')
def agronomist_dashboard():
    if session.get('role', '').lower() != 'agronomist':
        flash("Unauthorized access.", 'error')
        return redirect(url_for('logout'))
    return render_template('agronomist_dashboard.html',
                           username=session.get('username', 'Agronomist'),
                           location=session.get('location', ''))

# ---------------- Crop Health Prediction ----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict_crop():
    if 'username' not in session:
        flash('Please login first.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if trained_model is None:
            flash("Model not loaded.", 'error')
            return redirect(url_for('predict_crop'))

        file = request.files.get('crop_image')
        if not file or file.filename == '':
            flash("No file uploaded!", 'error')
            return redirect(url_for('predict_crop'))

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            # Load & preprocess image
            img = Image.open(file_path).convert("RGB")
            img = img.resize((150, 150))   # ✅ must match model training input size
            img_array = np.array(img, dtype=np.float32) / 255.0
            x = np.expand_dims(img_array, axis=0)

            # ✅ Use trained_model instead of crop_health_model
            pred = trained_model.predict(x)[0][0]   # sigmoid output → probability
            print("Prediction probability:", pred)

            # Interpret result
            health_status = "Healthy" if pred >= 0.5 else "Unhealthy"

            # Save to session
            session["last_crop_health"] = health_status

            return render_template(
                'crop_result.html',
                health=health_status,
                probability=round(float(pred), 2),
                uploaded_image=url_for('static', filename=f'uploads/{filename}')
            )

        except Exception as e:
            print("Prediction error:", e)
            flash(f"Error processing image: {str(e)}", 'error')
            return redirect(url_for('predict_crop'))

    return render_template('predict.html')


@app.route('/soil_prediction', methods=['GET', 'POST'])
def soil_prediction():
    if 'username' not in session:
        flash('Please login first.', 'error')
        return redirect(url_for('login'))

    prediction = None
    crop = None

    if request.method == 'POST':
        crop = request.form.get('crop')
        if not crop:
            flash("Please enter a crop name.", "error")
            return redirect(url_for('soil_prediction'))

        try:
            crop = crop.strip().capitalize()   # normalize input
            if crop in crop_soil_mapping:
                prediction = crop_soil_mapping[crop]
            else:
                prediction = "❌ No soil data available for this crop"
        except Exception as e:
            flash(f"Prediction failed: {e}", "error")

    return render_template("soil_prediction.html", crop=crop, prediction=prediction)


# ---------------- Query Form ----------------
@app.route('/query', methods=['GET', 'POST'])
def query_form():
    if 'username' not in session:
        flash('Please login first.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if not question:
            flash("Please enter a valid query ❌", "error")
            return redirect(url_for('query_form'))

        new_query = Query(username=session['username'], question=question)
        db.session.add(new_query)
        db.session.commit()

        flash("✅ Your query has been submitted successfully!", "success")
        return redirect(url_for('farmer_dashboard'))

    return render_template('query_form.html')

# ---------------- Live Weather ----------------
@app.route('/weather', methods=['GET', 'POST'])
def weather_page():
    weather_data = None
    if request.method == 'POST':
        city = request.form.get('city')
        API_KEY = "546bcf1a2803be0bfa9dab15e79ca03b"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather_data = {
                "city": city,
                "temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"].title(),
                "humidity": data["main"]["humidity"],
                "wind": data["wind"]["speed"]
            }
        else:
            flash("City not found, please try again ❌")
    return render_template('weather.html', weather=weather_data)
def get_weather_forecast(city: str):
    API_KEY = "546bcf1a2803be0bfa9dab15e79ca03b"
    forecast_data = []
    try:
        # Get lat/lon for the city
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        data = requests.get(url, timeout=8).json()
        lat, lon = data["coord"]["lat"], data["coord"]["lon"]

        # Fetch 7-day forecast
        url_forecast = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&appid={API_KEY}&units=metric"
        forecast = requests.get(url_forecast, timeout=8).json()

        for day in forecast["daily"][:7]:
            forecast_data.append({
                "date": datetime.fromtimestamp(day["dt"]).strftime("%a"),
                "temp": day["temp"]["day"],
                "icon": day["weather"][0]["icon"]
            })
    except Exception as e:
        print("❌ Forecast fetch failed:", e)
    return forecast_data


# ---------------- Logout ----------------
@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

# ---------------- API for Live Dashboard Data ----------------
@app.route('/api/dashboard-data')
def api_dashboard_data():
    username = session.get('username', 'Farmer')
    location = session.get('location', 'Unknown')
    role = session.get('role', '').lower()
    if role != 'farmer':
        return jsonify({"error": "Unauthorized"}), 403
    weather_alert, weather_status_class, weather_status_label = get_weather_alert(location)
    soil_recommendation = {"crop": "Wheat", "soil": "Loamy Soil", "fertilizer": "Urea (50kg/acre)"}
    crop_health = session.get("last_crop_health", "Not Analyzed Yet")
    last_query = Query.query.filter_by(username=username).order_by(Query.timestamp.desc()).first()
    query_status = last_query.answer if last_query else "No queries yet"
    return jsonify({
        "username": username,
        "location": location,
        "weather_alert": weather_alert,
        "weather_status_class": weather_status_class,
        "weather_status_label": weather_status_label,
        "soil_recommendation": soil_recommendation,
        "crop_health": crop_health,
        "query_status": query_status
    })

# ---------------- Run App ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)


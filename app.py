from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
import pickle
import requests
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from flask_apscheduler import APScheduler
from flask_migrate import Migrate
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# ---------------- App Setup ----------------

app = Flask(__name__)
app.secret_key = 'replace_with_a_secure_secret'   # replace in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(days=7)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

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
    mobile = db.Column(db.String(20), nullable=False, unique=True)  # New field

# ---------------- Query Model ----------------
class Query(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, default="Pending")
    status = db.Column(db.String(50), default="Pending")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
# ---------------- SMS Log Model ----------------
class SmsLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    to_number = db.Column(db.String(20), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text)
    success = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------- Soil Data Model ----------------
class SoilData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    ph = db.Column(db.Float)
    moisture = db.Column(db.Float)
    soil_type = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------- Crop Upload Model ----------------
class CropUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    crop_name = db.Column(db.String(150))
    prediction = db.Column(db.String(150))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
class CropHealth(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
# ---------------- Load Models ----------------
# Load Random Forest model
# Get project folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Random Forest model from project folder
rf_model_path = os.path.join(BASE_DIR, "rf_paddy_model.pkl")
rf_model = joblib.load(rf_model_path)

# Load ResNet50 as feature extractor
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# ---------------- Load soil health Model ----------------

# ---------------- Load Models ----------------
crop_model = joblib.load("crop_model.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")

API_KEY = "546bcf1a2803be0bfa9dab15e79ca03b"  # for getting location weather if needed



# ---------------- Fast2SMS config ----------------
import requests

FAST2SMS_API_KEY = "0vx4tmFCsT7yIk6B9NqhpYidMb8zfZulJHQnGWV2EPLXjoRDUc5eVNdO1GuMpaAym2liBFIhJtnjc7k"


# ---------------- Scheduler: daily weather alert with logging ----------------
def daily_weather_alert():
    with app.app_context():
        users = User.query.all()
        for u in users:
            if not u.mobile:
                continue

            try:
                weather_msg, _, _ = get_weather_alert(u.location)
            except Exception as e:
                weather_msg = f"Daily Weather: (error getting weather) - {e}"

            full_msg = f"Hello {u.username}, {weather_msg}"

            # Initialize error message
            error_msg = ""

            # Send SMS
            try:
                success = send_sms(u.mobile, full_msg)
                if not success:
                    error_msg = "Fast2SMS API returned failure"
            except Exception as e:
                print(f"❌ SMS sending failed for {u.mobile}: {e}")
                success = False
                error_msg = str(e)

            # Log SMS in DB safely
            sms_log = SmsLog(
                to_number=u.mobile,
                message=full_msg,
                success=success,
                response="Sent via Fast2SMS" if success else f"Failed: {error_msg}"
            )
            db.session.add(sms_log)

        db.session.commit()
        print("✅ Daily weather alerts processed and logged.")



# ---------------- Updated send_sms function ----------------
def send_sms(mobile: str, message: str):
    """
    Send SMS via Fast2SMS API.
    """
    try:
        url = "https://www.fast2sms.com/dev/bulkV2"
        payload = {
            "sender_id": "TXTIND",
            "message": message,
            "route": "v3",
            "numbers": mobile  # ✅ matches the variable name
        }
        headers = {
            "authorization": FAST2SMS_API_KEY,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        resp_json = response.json()
        print(f"✅ SMS sent to {mobile}: {resp_json}")
        return resp_json.get("return", False)  # True if success
    except Exception as e:
        print(f"❌ SMS send failed for {mobile}: {e}")
        return False


# ---------------- Helper: Live Weather Alert ----------------
OPENWEATHER_API_KEY = "546bcf1a2803be0bfa9dab15e79ca03b"  # <-- replace

def get_weather_alert(city: str):
    """
    Return a tuple: (message, status_class, status_label)
    """
    API_KEY = OPENWEATHER_API_KEY
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
    except Exception as e:
        return (f"Weather service error: {e}", "alert", "Error")

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
        mobile = request.form['mobile'].strip()  # mobile captured

        if role not in ['farmer', 'agronomist']:
            flash('Invalid role selected.', 'error')
            return redirect(url_for('signup'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('signup'))

        if User.query.filter_by(mobile=mobile).first():
            flash('Mobile number already registered.', 'error')
            return redirect(url_for('signup'))

        new_user = User(username=username, password=password, location=location, role=role, mobile=mobile)
        db.session.add(new_user)
        db.session.commit()

        # Save relevant info in session
        session['username'] = username
        session['location'] = location
        session['role'] = role
        session['mobile'] = mobile

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
            session['mobile'] =user.mobile
            session['role'] = user.role.strip().lower()
            flash('Login successful!', 'success')
            return redirect(url_for(f"{session['role']}_dashboard"))
        else:
            flash('Invalid credentials', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')




# ---------------- Farmer Dashboard ----------------
import random

CROP_PRICES = {
    "Wheat": "₹2200 kg",
    "Rice": "₹2800 kg",
    "Maize": "₹1900 kg",
    "Sugarcane": "₹300 kg",
    "Cotton": "₹6500 kg"
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

    analytics = Query.query.filter_by(username=session['username']) \
                           .order_by(Query.timestamp.desc()) \
                           .limit(5).all()

    farming_tip = random.choice(FARMING_TIPS)

    return render_template(
        'farmer_dashboard.html',
        username=session.get('username', 'Farmer'),
        location=session.get('location', ''),
        profile_image_url=url_for('static', filename='person1.jpg'),
        weather_alert=weather_alert,
        weather_status_class=status_class,
        weather_status_label=status_label,
        soil_prediction={"crop": "Wheat", "soil": "Loamy Soil", "fertilizer": "Urea (50kg/acre)"},
        crop_health=crop_health,
        query_status=query_status,
        query_list=query_list,
        analytics=analytics,
        crop_prices=CROP_PRICES,
        farming_tip=farming_tip
    )



# ----------------- AGRONOMIST DASHBOARD -----------------
@app.route('/agronomist_dashboard')
def agronomist_dashboard():
    if session.get('role', '').lower() != 'agronomist':
        flash("Unauthorized access.", 'error')
        return redirect(url_for('logout'))

    queries = Query.query.order_by(Query.timestamp.desc()).all()
    crop_data = CropHealth.query.order_by(CropHealth.timestamp.desc()).all()  # 👈 changed
    soil_data = SoilData.query.order_by(SoilData.timestamp.desc()).all()
    farmers = User.query.filter_by(role='farmer').all()

    return render_template('agronomist_dashboard.html',
                           username=session.get('username'),
                           location=session.get('location'),
                           queries=queries,
                           crop_data=crop_data,
                           soil_data=soil_data,
                           farmers=farmers)

# ----------------- EDIT CROP -----------------
@app.route('/edit_crop', methods=['POST'])
def edit_crop():
    if session.get('role', '').lower() != 'agronomist':
        flash("Unauthorized access.", 'error')
        return redirect(url_for('logout'))

    crop_id = request.form.get('crop_id')
    new_prediction = request.form.get('prediction')
    crop = CropUpload.query.get(crop_id)
    if crop:
        crop.prediction = new_prediction
        db.session.commit()
        flash("✅ Crop data updated successfully!", "success")
    else:
        flash("❌ Crop record not found.", "error")

    return redirect(url_for('agronomist_dashboard'))

@app.route('/update_crop_prediction', methods=['POST'])
def update_crop_prediction():
    if session.get('role', '').lower() != 'agronomist':
        flash("❌ Unauthorized access.", "error")
        return redirect(url_for('logout'))

    crop_id = request.form.get('crop_id')
    new_status = request.form.get('new_status')

    if not crop_id or not new_status:
        flash("⚠ Invalid update request.", "error")
        return redirect(url_for('agronomist_dashboard'))

    try:
        crop = CropHealth.query.get(int(crop_id))
        if crop:
            crop.prediction = new_status
            db.session.commit()
            flash(f"✅ Crop status updated to '{new_status}'", "success")
        else:
            flash("❌ Crop record not found.", "error")

    except Exception as e:
        db.session.rollback()
        flash(f"❌ Error updating crop: {str(e)}", "error")

    return redirect(url_for('agronomist_dashboard'))

# ----------------- EDIT SOIL DATA -----------------
@app.route('/edit_soil', methods=['POST'])
def edit_soil():
    if session.get('role', '').lower() != 'agronomist':
        flash("Unauthorized access.", 'error')
        return redirect(url_for('logout'))

    soil_id = request.form.get('soil_id')
    username = request.form.get('username')
    soil_type = request.form.get('soil_type')
    crop_type = request.form.get('crop_type')
    location = request.form.get('location')

    soil = SoilData.query.get(soil_id)
    if soil:
        soil.username = username
        soil.soil_type = soil_type
        soil.crop_type = crop_type
        soil.location = location
        db.session.commit()
        flash("✅ Soil data updated successfully!", "success")
    else:
        flash("❌ Soil record not found.", "error")

    return redirect(url_for('agronomist_dashboard'))


# ----------------- REPLY TO QUERY -----------------
@app.route('/reply_query', methods=['POST'])
def reply_query():
    if session.get('role', '').lower() != 'agronomist':
        flash("❌ Unauthorized access.", "error")
        return redirect(url_for('logout'))

    query_id = request.form.get("id")
    reply_text = request.form.get("reply_text", "").strip()

    if not query_id or not reply_text:
        flash("❌ Query ID or reply text missing.", "error")
        return redirect(url_for("agronomist_dashboard"))

    query = Query.query.get(query_id)
    if not query:
        flash("❌ Query not found.", "error")
        return redirect(url_for("agronomist_dashboard"))

    query.answer = reply_text
    query.status = "Answered"
    db.session.commit()

    flash("✅ Reply sent successfully!", "success")
    return redirect(url_for("agronomist_dashboard"))
# ---------------- Helper Function ----------------
def predict_leaf(img_path):
    """
    Predicts the health of a paddy crop image using ResNet50 + Random Forest.
    """
    # Preprocess image
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features
    features = resnet_model.predict(img_array, verbose=0)

    # Predict with RF model
    pred = rf_model.predict(features)[0]

    if pred == 0:
        return "Healthy"
    else:
        return "Unhealthy"

# ---------------- Crop Health Prediction Route ----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict_crop():
    if 'username' not in session:
        flash('Please login first.', 'error')
        return redirect(url_for('login'))

    if rf_model is None:
        flash("Model not loaded.", 'error')
        return redirect(url_for('predict_crop'))

    if request.method == 'POST':
        file = request.files.get('crop_image')
        if not file or file.filename == '':
            flash("No file uploaded!", 'error')
            return redirect(url_for('predict_crop'))

        try:
            filename = secure_filename(file.filename)
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            # Predict crop health with the file path inside POST block
            main_status = predict_leaf(file_path)
            session["last_crop_health"] = main_status

            # Save result in database
            new_record = CropHealth(
                username=session['username'],
                image_path=f'uploads/{filename}',
                prediction=main_status
            )
            db.session.add(new_record)
            db.session.commit()

            return render_template(
                'crop_result.html',
                health=main_status,
                uploaded_image=url_for('static', filename=f'uploads/{filename}')
            )

        except Exception as e:
            print("Prediction error:", e)
            flash(f"Error processing image: {str(e)}", 'error')
            return redirect(url_for('predict_crop'))

    return render_template('predict.html')


# ---------------- Soil Prediction Route ----------------
@app.route('/soil_prediction', methods=['GET', 'POST'])
def soil_prediction():
    soil = None
    location = None
    weather = None
    recommendation = None

    if request.method == 'POST':
        soil = request.form.get("soil")
        location = request.form.get("location")

        if not soil or not location:
            flash("Please enter both soil type and location!", "error")
            return redirect(url_for("soil_prediction"))

        # Fetch weather
        weather, error = get_weather_for_soil(location)
        if error:
            flash(f"Weather API error: {error}", "error")
            return redirect(url_for("soil_prediction"))

        # Dummy nutrient values
        nitrogen = 80
        phosphorus = 40
        potassium = 40
        fertilizer = 100

        # If model expects encoded soil
        soil_encoded = soil_encoder.transform([soil])[0]

        features = np.array([[weather["temp"], weather["rainfall"], fertilizer,
                              nitrogen, phosphorus, potassium, soil_encoded]])

        prediction = crop_model.predict(features)[0]
        recommendation = f"Recommended Crop: {prediction}"

    return render_template("soil_prediction.html",
                           soil=soil,
                           location=location,
                           weather=weather,
                           recommendation=recommendation)



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

# ---------------- Live Weather Page ----------------
@app.route('/weather', methods=['GET', 'POST'])
def weather_page():
    weather_data = None
    if request.method == 'POST':
        city = request.form.get('city')
        API_KEY = OPENWEATHER_API_KEY
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
    API_KEY = OPENWEATHER_API_KEY
    forecast_data = []
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        data = requests.get(url, timeout=8).json()
        lat, lon = data["coord"]["lat"], data["coord"]["lon"]
        url_forecast = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&appid={API_KEY}&units=metric"
        forecast = requests.get(url_forecast, timeout=8).json()
        for day in forecast.get("daily", [])[:7]:
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
@app.route('/fetch_crop_images')
def fetch_crop_images():
    """Return JSON of all crop health images"""
    try:
        images = CropHealth.query.order_by(CropHealth.timestamp.desc()).all()
        data = [
            {
                "id": img.id,
                "username": img.username,
                "image_path": img.image_path,
                "prediction": img.prediction,
                "timestamp": img.timestamp.strftime("%Y-%m-%d %H:%M")
            }
            for img in images
        ]
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ---------------- Run App ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Recreate tables with new columns
    app.run(debug=True, use_reloader=False)





from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from datetime import timedelta

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
    role = db.Column(db.String(50), nullable=False)  # farmer or agronomist

# Load Crop Health Model
try:
    crop_health_model = load_model('crop_health_model.h5')
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model load failed:", e)
    crop_health_model = None

# ---------------- Home Page ----------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------- Signup Page ----------------
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

        new_user = User(username=username, password=password,
                        location=location, role=role)
        db.session.add(new_user)
        db.session.commit()

        session['username'] = username
        session['location'] = location
        session['role'] = role
        flash('Signup successful!', 'success')

        return redirect(url_for(f"{role}_dashboard"))

    return render_template('signup.html')

# ---------------- Login Page ----------------
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
@app.route('/farmer')
def farmer_dashboard():
    if session.get('role', '').lower() == 'farmer':
        return render_template('farmer_dashboard.html',
                               username=session['username'],
                               location=session['location'])
    flash("Unauthorized access.", 'error')
    return redirect(url_for('logout'))

# ---------------- Agronomist Dashboard ----------------
@app.route('/agronomist')
def agronomist_dashboard():
    if session.get('role', '').lower() == 'agronomist':
        return render_template('agronomist_dashboard.html',
                               username=session['username'],
                               location=session['location'])
    flash("Unauthorized access.", 'error')
    return redirect(url_for('logout'))

# ---------------- Crop Health Prediction ----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict_crop():
    if 'username' not in session:
        flash('Please login first.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if crop_health_model is None:
            flash("Model not loaded. Please contact admin.", 'error')
            return redirect(url_for('predict_crop'))

        file = request.files.get('crop_image')
        img_data = request.form.get('crop_image')
        img_base64 = None  # to send to template

        if file:
            img = Image.open(file)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
        elif img_data:
            try:
                header, encoded = img_data.split(",", 1)
                decoded = base64.b64decode(encoded)
                img = Image.open(BytesIO(decoded))
                img_base64 = img_data  # already base64 string
            except:
                flash("Invalid image data!", 'error')
                return redirect(url_for('predict_crop'))
        else:
            flash("No image provided!", 'error')
            return redirect(url_for('predict_crop'))

        img = img.resize((150,150))
        x = np.array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        pred = crop_health_model.predict(x)
        health_status = 'Healthy' if pred[0][0] > 0.5 else 'Unhealthy'

        return render_template('crop_result.html', health=health_status, image_url=img_base64)

    return render_template('predict.html')


# ---------------- Logout ----------------
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ---------------- Run App ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

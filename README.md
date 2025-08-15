# 🌾 AgroScan – Smart Farming Advisory & Crop Health Classification

AgroScan is a smart agricultural web application designed to assist farmers and agronomists in improving crop productivity. The platform features **two user portals** – one for **Farmers** and another for **Agronomists/Admins** – and utilizes **image processing and machine learning** to detect crop health through leaf images.

---

## 🚀 Features

### 👨‍🌾 Farmer Portal
- Upload crop/leaf images for health check
- Get predictions: **Healthy** or **Unhealthy**
- View suggestions for treatment (if unhealthy)
- Simple and mobile-friendly UI

### 👨‍🔬 Agronomist/Admin Portal
- View all farmer submissions
- Add treatment suggestions for unhealthy crops
- Dashboard with crop health analytics and reports

### 🤖 Crop Health Classification
- Trained CNN model using plant disease dataset
- Image processing and deep learning (Keras + TensorFlow)
- Predicts based on leaf images

---

## 🛠️ Tech Stack

| Layer         | Technology Used                    |
|---------------|------------------------------------|
| Frontend      | HTML, CSS, JavaScript              |
| Backend       | Python, Flask                      |
| ML/AI         | TensorFlow, Keras, OpenCV          |
| Database      | SQLite / Oracle                    |
| Visualization | Chart.js / Plotly.js               |
| Deployment    | (Optional) Render / Heroku         |

---

## 📁 Folder Structure

AgroScan/
├── app.py
├── farmer_portal/
│ └── routes.py
├── agronomist_portal/
│ └── routes.py
├── model/
│ ├── crop_health_model.h5
│ └── predict.py
├── templates/
│ ├── upload.html
│ └── result.html
├── static/
│ └── uploads/
├── README.md
└── requirements.txt


---

## 📷 Sample Prediction Flow

1. User uploads crop image
2. ML model processes image
3. Predicts: **Healthy / Unhealthy**
4. Suggestion displayed (if needed)

---

## ✅ How to Run Locally

```bash
git clone https://github.com/Harshini-rnk04/AgroScan.git
cd AgroScan
pip install -r requirements.txt
python app.py

Open in browser: http://127.0.0.1:5000

📌 Status
🚧 Under Development — Mini Project (2025)
🎓 Built for academic learning & practical ML deployment

👩‍💻 Contributors
Harshini R.N.K  – ML & Web Development

Ekaashari R.R– Dashboard & Admin Portal


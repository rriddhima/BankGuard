# 🛡️ BankGuard — AI-Powered Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-4ade80?style=flat-square)

A full-stack machine learning project that detects fraudulent bank transactions in real time. Built with a Random Forest classifier, a Flask REST API, and an interactive dashboard UI.

---

## 📸 Demo

> Enter a transaction → get an instant fraud prediction with risk score, probability bar, and live-updating charts.

| Dashboard Overview | Fraud Alert |
|---|---|
| *<img width="1885" height="964" alt="image" src="https://github.com/user-attachments/assets/fffd0b9d-3833-4335-909b-cebd28295e5b" />
* | *<img width="1887" height="911" alt="image" src="https://github.com/user-attachments/assets/3e486f98-25f3-41ad-a8f1-e41ba0ab190e" />
* |

---

## 🧠 How It Works

```
Transaction Data  →  Flask API  →  ML Model  →  Fraud Score  →  Dashboard
```

1. The **Random Forest model** is trained on 284,807 real credit card transactions (Kaggle dataset)
2. The **Flask API** loads the trained model and exposes a `/predict` endpoint
3. The **Dashboard** sends transaction data to the API and visualizes the result in real time

---

## 🗂️ Project Structure

```
bankguard/
├── ml/
│   ├── train_model.py          # Train & save the ML model
│   ├── confusion_matrix.png    # Model evaluation chart
│   ├── feature_importance.png  # Feature analysis chart
│   └── data/
│       └── creditcard.csv      # Dataset (download separately)
├── api/
│   ├── app.py                  # Flask REST API
│   ├── model.pkl               # Saved trained model
│   ├── scaler.pkl              # Saved scaler
│   └── requirements.txt
├── frontend/
│   └── index.html              # Dashboard UI
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/bankguard.git
cd bankguard
```

### 2. Install dependencies

```bash
cd api
pip install -r requirements.txt
```

### 3. Download the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:

```
ml/data/creditcard.csv
```

> **No Kaggle account?** The training script auto-generates sample data so you can still run the full project.

### 4. Train the model

```bash
cd ml
python train_model.py
```

This will:
- Train a Random Forest classifier
- Save `model.pkl` and `scaler.pkl` to the `api/` folder
- Generate evaluation charts

### 5. Start the API

```bash
cd api
python app.py
```

API runs at `http://localhost:5000`

### 6. Open the dashboard

Open `frontend/index.html` in your browser. That's it!

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Analyze a transaction |
| GET | `/transactions` | Get transaction history |
| GET | `/stats` | Get summary statistics |

### Example Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 4500, "merchant": "Amazon", "category": "Shopping"}'
```

### Example Response

```json
{
  "transaction_id": "TXN482931",
  "amount": 4500.0,
  "is_fraud": false,
  "fraud_probability": 3.21,
  "risk_level": "LOW",
  "merchant": "Amazon",
  "category": "Shopping",
  "timestamp": "2026-03-26 10:30:00"
}
```

---

## 📊 Model Performance

| Metric | Legitimate | Fraud |
|--------|-----------|-------|
| Precision | 99% | 93% |
| Recall | 99% | 82% |
| F1 Score | 99% | 87% |

- **Algorithm:** Random Forest (100 estimators)
- **Dataset:** 284,807 transactions, 0.17% fraud rate
- **Handling imbalance:** `class_weight='balanced'`

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | scikit-learn, pandas, numpy |
| Backend API | Flask, flask-cors |
| Model Persistence | joblib |
| Frontend | HTML, CSS, JavaScript |
| Charts | Chart.js |
| Dataset | Kaggle Credit Card Fraud Detection |

---

## 🚀 Future Improvements

- [ ] PostgreSQL database for persistent transaction storage
- [ ] User authentication and role-based access
- [ ] Real-time alerts via email/SMS
- [ ] Docker containerization
- [ ] Deploy to AWS / Render

---

## 👤 Author

**RIDDHIMA**

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

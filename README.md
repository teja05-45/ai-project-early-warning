📌 AI Early-Warning System for Software Project Delivery Risk
End-to-End Machine Learning & Decision Intelligence Project

A full end-to-end AI system that predicts software project delivery failure risk before it happens, explains why the risk is increasing, tracks risk trends across sprints, and allows managers to simulate corrective actions.

Designed to demonstrate real-world Applied ML + Product Engineering skills, including:

Sprint-level data modeling

Feature engineering from engineering & team metrics

Risk prediction using ML

Explainable AI (SHAP)

Risk trend analysis & early warnings

What-if simulation engine

Executive-friendly Streamlit dashboard

🚀 Live Demo

👉 Open the deployed Streamlit app: https://ai-project-early-warning.streamlit.app/

📁 Project Structure

```
ai-project-early-warning/
│
├── data/
│   └── raw/
│       └── project_delivery_risk.csv   # Sprint-level dataset
│
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py          # Risk signal generation
│   ├── train_model.py                  # Model training & saving
│   ├── simulation.py                   # What-if simulation engine
│   └── explainability.py               # SHAP explainability logic
│
├── dashboard/
│   └── app.py                          # Streamlit decision dashboard
│
├── models/
│   ├── delivery_risk_model.joblib      # Trained ML model
│   └── feature_columns.joblib          # Model feature list
│
├── requirements.txt
├── README.md
└── .gitignore
```

🧠 Key Features
✔ End-to-End AI Risk Pipeline

Processes sprint-level engineering & team metrics

Engineers delivery risk signals

Trains a probabilistic ML risk model

Outputs risk score (0–100%)

✔ Explainable AI (SHAP)

Identifies top drivers behind delivery risk

Explains predictions in business-friendly language

Supports:

Root-cause analysis

Driver change vs previous sprint

✔ Risk Trends & Early-Warning Alerts

Tracks risk evolution across sprints

Detects:

🔺 Risk escalation

🔻 Risk improvement

➖ Stable risk

Flags early-warning signals before failure

✔ What-If Simulation Engine

Managers can simulate:

Reducing scope

Resolving dependency blockers

Reducing overtime

Improving quality

…and instantly see how delivery risk changes, without retraining the model.

✔ Executive-Ready Streamlit Dashboard

Portfolio-level risk visibility

Project deep-dive view

Risk buckets (Low / Medium / High / Critical)

Root-cause explanations

Actionable recommendations

🧪 Key Features Used for Prediction

Sprint velocity & backlog growth

Scope change requests

Dependency blockers

Bug count & reopened issues

Code churn & test coverage

Team workload & attrition risk

Communication delays

Milestone slippage

These features reflect real delivery signals used in industry, not toy examples.

🛠️ Tech Stack
| Layer           | Technology                       |
| --------------- | -------------------------------- |
| Language        | Python                           |
| ML Model        | Gradient Boosting (scikit-learn) |
| Explainability  | SHAP                             |
| Data Processing | Pandas, NumPy                    |
| Dashboard       | Streamlit                        |
| Model Storage   | Joblib                           |
| Visualization   | Streamlit charts                 |

📦 Installation

Clone the repository:

git clone https://github.com/teja05-45/ai-project-early-warning.git
cd ai-project-early-warning


Create virtual environment:

python -m venv venv


Activate it:

Windows

venv\Scripts\activate


macOS / Linux

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

🤖 Train the Model
python src/train_model.py


This will generate:

Trained delivery risk model

Feature column metadata

Ready-to-use inference artifacts

🌐 Run Streamlit App
streamlit run dashboard/app.py

App Pages
| Page              | Description                     |
| ----------------- | ------------------------------- |
| Executive Summary | Portfolio-level risk overview   |
| Project View      | Sprint-level deep dive          |
| Risk Trends       | Risk direction & early warnings |
| Risk Explanation  | SHAP-based root cause analysis  |
| What-If Simulator | Decision simulation             |


🧭 Future Enhancements

Prediction confidence / uncertainty score

Portfolio risk heatmap

Automated alerts (Slack / Email)

Time-to-failure estimation

Integration with Jira / Azure DevOps APIs

XGBoost / LightGBM models

👨‍💻 Author

Teja Matta
Machine Learning & Applied AI Engineer

GitHub: https://github.com/teja05-45

LinkedIn: https://www.linkedin.com/in/teja-matta-602b3531a



# ScamShield

ScamShield is a full-stack application that uses machine learning to identify potentially fraudulent job advertisements.

The project uses TF-IDF and Logistic Regression for text classification, with a FastAPI backend, React frontend, and MongoDB Atlas for storing analysis history.

## Tech Stack

- **Frontend:** React, TypeScript, Vite
- **Backend:** Python, FastAPI
- **ML:** Scikit-learn, TF-IDF, Logistic Regression
- **Database:** MongoDB Atlas
- **Model Storage:** Joblib

## How It Works

1. Enter a job advertisement in the application.
2. The backend processes the text using the trained ML pipeline.
3. The model predicts whether the listing may be fraudulent.
4. The application displays the result and risk score.
5. The analysis is saved to MongoDB Atlas for later reference.

## Model Evaluation

The model was evaluated on a held-out test set.

| Metric | Score |
|---|---:|
| Precision | 65.81% |
| Recall | 89.02% |
| F1-Score | 75.68% |

These results reflect performance on the evaluation dataset. Predictions are estimates and should not be treated as proof that a job advertisement is fraudulent.

## Project Structure

```text
ScamShield/
├── app/
│   └── main.py
├── artifacts/
│   └── scamshield_model.joblib
├── data/
│   └── processed/
├── frontend/
│   ├── public/
│   └── src/
├── scripts/
│   ├── analyze_dataset.py
│   ├── explore_dataset.py
│   └── train_model.py
├── requirements.txt
└── test_mongodb.py
```

## Running Locally

### Backend

Create a `.env` file in the project root with the MongoDB connection string expected by the application.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`

### Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite.

<img width="1883" height="896" alt="image" src="https://github.com/user-attachments/assets/ada23429-c63b-4986-ac42-79397bfc74fa" />
<img width="1888" height="908" alt="image" src="https://github.com/user-attachments/assets/461c0fda-aee6-4d24-b151-ebb2da571b94" />
<img width="1918" height="926" alt="image" src="https://github.com/user-attachments/assets/8ed02341-5566-44bc-aab3-7c134a9a91dc" />
<img width="1917" height="902" alt="image" src="https://github.com/user-attachments/assets/2e44e792-82f4-47c0-9ca9-a763f673c7e0" />

## Notes

- The model's score is not a calibrated probability of fraud.
- Results may vary on job advertisements outside the evaluation dataset.
- Database credentials should be kept in environment variables and never committed to Git.

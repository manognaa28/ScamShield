import os
from pathlib import Path
from datetime import datetime, timezone

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import PyMongoError


app = FastAPI(
    title="ScamShield API",
    description=(
        "Paste job-ad text into POST /predict to receive an estimated "
        "fraud-risk score. This is an educational tool, not a fraud verdict."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# Paths are based on this file so the API works no matter where you start uvicorn.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "scamshield_model.joblib"
# Load environment variables from the project root.
load_dotenv(PROJECT_ROOT / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "scamshield")

mongo_client = None
analyses_collection = None

if MONGODB_URI:
    try:
        mongo_client = MongoClient(
            MONGODB_URI.strip(),
            serverSelectionTimeoutMS=5000,
        )

        # Verify that Atlas is reachable during startup.
        mongo_client.admin.command("ping")

        database = mongo_client[DATABASE_NAME]
        analyses_collection = database["analyses"]

        print("ScamShield: MongoDB Atlas connected successfully.")

    except PyMongoError as error:
        print(
            "ScamShield: MongoDB connection failed. "
            f"Analysis history will be unavailable: {type(error).__name__}"
        )
        if mongo_client is not None:
            mongo_client.close()
        mongo_client = None
        analyses_collection = None

else:
    print(
        "ScamShield: MONGODB_URI is missing. "
        "Analysis history will be unavailable."
    )


# Must match training: title, company_profile, description, requirements, benefits.
TEXT_FIELDS = (
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
)


# Provisional display thresholds on the 0-100 risk_score. These are not
# validated real-world risk boundaries.
LOW_SCORE_MAX = 30.0
MODERATE_SCORE_MAX = 70.0

DISCLAIMER = (
    "This result is an estimate based on historical job advertisements "
    "(EMSCAD), not proof that a posting is fraudulent or legitimate. "
    "The risk score is an uncalibrated model score from logistic regression, "
    "not a verified real-world probability. Risk levels use provisional "
    "display thresholds (Low below 30, Moderate from 30 to below 70, "
    "High at 70 and above), not validated real-world risk boundaries."
)


def load_model():
    """Load the saved sklearn Pipeline once. Fail clearly if the file is missing."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "ScamShield model file was not found at:\n"
            f"  {MODEL_PATH}\n"
            "Train the model first with: python scripts/train_model.py\n"
            "The API will not start until that file exists."
        )

    try:
        return joblib.load(MODEL_PATH)
    except Exception as error:
        raise RuntimeError(
            f"Could not load the ScamShield model from {MODEL_PATH}: {error}"
        ) from error


# Loaded once when this module is imported (API startup). Not reloaded per request.
model = load_model()


def fraudulent_class_index(loaded_model) -> int:
    """Find the column in predict_proba that belongs to class 1 (fraudulent)."""
    classes = getattr(loaded_model, "classes_", None)

    if classes is None:
        raise RuntimeError("The loaded model has no classes_ attribute.")

    class_list = list(classes)

    if 1 not in class_list:
        raise RuntimeError(
            "The loaded model does not include class 1 (fraudulent). "
            f"Found classes_: {class_list}"
        )

    return class_list.index(1)


FRAUDULENT_INDEX = fraudulent_class_index(model)


class JobAdRequest(BaseModel):
    title: str = Field(default="", max_length=300)
    company_profile: str = Field(default="", max_length=20000)
    description: str = Field(default="", max_length=50000)
    requirements: str = Field(default="", max_length=20000)
    benefits: str = Field(default="", max_length=20000)


class PredictionResponse(BaseModel):
    prediction: str
    risk_score: float
    risk_level: str
    disclaimer: str
    risk_level_note: str
    analysis_id: str | None = None
    created_at: datetime | None = None

class HistoryItem(BaseModel):
    id: str
    title: str
    company_profile: str
    description: str
    requirements: str
    benefits: str
    prediction: str
    risk_score: float
    risk_level: str
    created_at: datetime


def combine_text(payload: JobAdRequest) -> str:
    """Join fields with spaces, same order and missing-value handling as training."""
    parts = []

    for name in TEXT_FIELDS:
        value = getattr(payload, name)

        if value is None:
            value = ""

        parts.append(str(value))

    return " ".join(parts)


def risk_level_from_score(score: float) -> str:
    if score < LOW_SCORE_MAX:
        return "Low"

    if score < MODERATE_SCORE_MAX:
        return "Moderate"

    return "High"


@app.get("/")
def root():
    return {
        "name": "ScamShield API",
        "summary": (
            "Paste job-ad text into POST /predict to receive an estimated "
            "fraud-risk score. This is an educational tool, not a fraud verdict."
        ),
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }

@app.get("/history", response_model=list[HistoryItem])
def get_history(limit: int = 20):
    """Return the most recent saved job-ad analyses."""
    if analyses_collection is None:
        raise HTTPException(
            status_code=503,
            detail="Analysis history is unavailable. Check the MongoDB connection.",
        )

    limit = max(1, min(limit, 100))

    try:
        documents = analyses_collection.find(
            {},
            {
                "title": 1,
                "company_profile": 1,
                "description": 1,
                "requirements": 1,
                "benefits": 1,
                "prediction": 1,
                "risk_score": 1,
                "risk_level": 1,
                "created_at": 1,
            },
        ).sort("created_at", -1).limit(limit)

        history = []

        for document in documents:
            history.append(
                {
                    "id": str(document["_id"]),
                    "title": document.get("title", ""),
                    "company_profile": document.get("company_profile", ""),
                    "description": document.get("description", ""),
                    "requirements": document.get("requirements", ""),
                    "benefits": document.get("benefits", ""),
                    "prediction": document["prediction"],
                    "risk_score": document["risk_score"],
                    "risk_level": document["risk_level"],
                    "created_at": document["created_at"],
                }
            )

        return history

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve analysis history. Please try again.",
        )

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: JobAdRequest):
    # Collect the job advertisement fields.
    fields = [
        payload.title,
        payload.company_profile,
        payload.description,
        payload.requirements,
        payload.benefits,
    ]

    # Require both a job title and a job description.
    if not payload.title.strip():
        raise HTTPException(
            status_code=422,
            detail="Please enter a job title before analyzing the advertisement.",
        )

    if not payload.description.strip():
        raise HTTPException(
            status_code=422,
            detail="Please enter a job description before analyzing the advertisement.",
        )

    # The other three fields remain optional.
    # in_balanced_dataset is intentionally never accepted or scored.
    combined_text = combine_text(payload)

    # Run the ML model.
    try:
        predicted_label = int(model.predict([combined_text])[0])
        probabilities = model.predict_proba([combined_text])[0]
        fraudulent_score = float(probabilities[FRAUDULENT_INDEX])

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The model could not score this posting. Please try again.",
        )

    # Calculate the risk score and classification.
    risk_score = round(fraudulent_score * 100.0, 2)
    prediction = "fraudulent" if predicted_label == 1 else "legitimate"
    risk_level = risk_level_from_score(risk_score)
    created_at = datetime.now(timezone.utc)

    # Check whether MongoDB is available.
    if analyses_collection is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The prediction was calculated, but could not be saved "
                "because analysis history is unavailable. Please try again later."
            ),
        )

    # Save the analysis to MongoDB Atlas.
    try:
        result = analyses_collection.insert_one(
            {
                "title": payload.title,
                "company_profile": payload.company_profile,
                "description": payload.description,
                "requirements": payload.requirements,
                "benefits": payload.benefits,
                "prediction": prediction,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "created_at": created_at,
            }
        )

        analysis_id = str(result.inserted_id)

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail=(
                "The prediction was calculated, but could not be saved "
                "to the analysis history. Please try again."
            ),
        )

    # Return the prediction and saved analysis details.
    return PredictionResponse(
        prediction=prediction,
        risk_score=risk_score,
        risk_level=risk_level,
        disclaimer=DISCLAIMER,
        risk_level_note=(
            "risk_level uses provisional display thresholds: "
            "Low < 30, Moderate 30 to < 70, High >= 70. "
            "These are not validated real-world risk boundaries."
        ),
        analysis_id=analysis_id,
        created_at=created_at,
    )
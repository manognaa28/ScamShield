import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# Load environment variables from the project root .env file.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "scamshield")

if not MONGODB_URI:
    raise RuntimeError(
        "MONGODB_URI was not found. Check that your .env file "
        "is in the project root and contains MONGODB_URI."
    )

client = MongoClient(
    MONGODB_URI,
    server_api=ServerApi("1"),
    serverSelectionTimeoutMS=15000,
)

try:
    # Verify that Python can reach MongoDB Atlas.
    client.admin.command("ping")
    print("SUCCESS: Connected to MongoDB Atlas!")

    # Select the ScamShield database.
    database = client[DATABASE_NAME]

    # Create a temporary test document.
    collection = database["connection_tests"]
    result = collection.insert_one(
        {
            "message": "ScamShield MongoDB connection test",
            "status": "success",
        }
    )

    print("SUCCESS: Inserted a test document.")
    print(f"Database: {DATABASE_NAME}")
    print(f"Test document ID: {result.inserted_id}")

    # Read the test document back.
    saved_document = collection.find_one({"_id": result.inserted_id})

    if saved_document:
        print("SUCCESS: Read the test document back from MongoDB.")

    # Remove the temporary test document.
    collection.delete_one({"_id": result.inserted_id})
    print("SUCCESS: Removed the temporary test document.")

except Exception as error:
    print("MONGODB CONNECTION TEST FAILED")
    print(f"Error type: {type(error).__name__}")
    print(f"Error details: {error}")

finally:
    client.close()
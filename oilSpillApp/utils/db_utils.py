import os
import base64
import io
from PIL import Image
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

# --- Load environment variables ---
load_dotenv()

# --- Cached MongoDB connection ---
@st.cache_resource
def get_db_connection():
    mongo_uri = st.secrets.get("mongo", {}).get("uri") if "mongo" in st.secrets else os.getenv("MONGO_URI")
    mongo_db = st.secrets.get("mongo", {}).get("db") if "mongo" in st.secrets else os.getenv("MONGO_DB")

    print("🔍 [DEBUG] MONGO_URI =", mongo_uri)
    print("🔍 [DEBUG] MONGO_DB =", mongo_db)

    if not mongo_uri or not mongo_db:
        raise ValueError("Missing MongoDB environment variables or secrets")

    client = MongoClient(mongo_uri)
    db = client[mongo_db]

    detections_collection = db["detections"]
    users_collection = db["users"]

    print("✅ Connected to MongoDB successfully!")
    return detections_collection, users_collection


# Create global connection once
detections_collection, users_collection = get_db_connection()


# --- Helper functions ---
def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))


# --- Database operations ---
def save_result_to_db(username, filename, original_img, mask_img, metrics):
    record = {
        "username": username,
        "filename": filename,
        "original_image": image_to_base64(original_img),
        "mask_image": image_to_base64(mask_img),
        "metrics": metrics,
        "timestamp": datetime.utcnow()
    }
    detections_collection.insert_one(record)


from gridfs import GridFS

def get_user_history(username):
    """Fetch all detection results for a user"""
    db = detections_collection.database
    fs = GridFS(db)

    results = detections_collection.find({"user": username}).sort("timestamp", -1)
    history = []
    for r in results:
        original_img = fs.get(r["original_image_id"]).read()
        mask_img = fs.get(r["mask_image_id"]).read()

        history.append({
            "filename": r["filename"],
            "timestamp": r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "original": Image.open(io.BytesIO(original_img)),
            "mask": Image.open(io.BytesIO(mask_img)),
        })
    return history
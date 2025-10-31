import streamlit as st
from PIL import Image
import numpy as np
import io
import gdown
import os
import tensorflow as tf 
from utils.db_utils import save_result_to_db, get_user_history
from tensorflow.keras import layers, Model, backend as K
from dotenv import load_dotenv
import gridfs
import certifi
from pymongo import MongoClient
import bcrypt
import datetime

st.markdown("""
    <style>
    <!-- Load Google Material Icons -->
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    
    /* Background gradient for the whole app */
    .stApp {
        background: linear-gradient(135deg, #e0f2ff 0%, #cce7ff 50%, #99d4ff 100%);
        background-attachment: fixed;
    }

    /* Full-width blue title bar */
    .page-title {
        background-color: #064e7b;
        color: white;
        padding: 16px 25px;
        border-radius: 8px;
        font-size: 28px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 25px;
        width: 100%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Subtitles under sections */
    .section-subtitle {
        background-color: #0056b3;
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 20px;
        font-weight: 500;
        margin-top: 15px;
        text-align: center;
    }

    /* --- Main content layout --- */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 80%;              /* Shrinks width to make it centered */
        margin: auto !important;     /* Centers the content */
        border-radius: 12px;         /* rounded edges */
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .page-title .material-icons {
        font-size: 36px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def page_header(icon_name, title_text):
    st.markdown(f"""
        <div class="page-title">
            <span class="material-symbols-outlined">{icon_name}</span>
            {title_text}
        </div>
    """, unsafe_allow_html=True)

# Load environment variables
# load_dotenv()

# MongoDB setup
try:
    client = MongoClient(st.secrets["mongo"]["uri"], tlsCAFile=certifi.where())
    db = client["oilspill"]
    # Initialize GridFS for storing images
    fs = gridfs.GridFS(db)
except Exception as e:
    st.error(f"MongoDB connection failed: {e}")
    
users_collection = db["users"]
collection = db["detections"]   # collection for storing detection history



def register_user(username, password):
    # Check if user exists
    if users_collection.find_one({"username": username}):
        st.error("Username already exists.")
        return False

    # Hash password
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Store in MongoDB
    users_collection.insert_one({"username": username, "password": hashed_pw})
    st.success("Registration successful!")
    return True

def login_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user["password"]):
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        st.success(f"Welcome, {username}!")
        return True
    else:
        st.error("Invalid username or password.")
        return False


# ---------- Custom Metrics ----------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(tf.cast(y_pred > 0.5, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(tf.cast(y_pred > 0.5, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def soft_iou(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# ---------- Custom Loss ----------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def weighted_bce(y_true, y_pred, beta=3.0):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reshape(bce, [-1])
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    weights = 1 + (beta - 1) * y_true_f
    return K.mean(weights * bce)

def combo_loss(y_true, y_pred, alpha=0.5, beta=3.0):
    return alpha * dice_loss(y_true, y_pred) + (1 - alpha) * weighted_bce(y_true, y_pred, beta=beta)

def chosen_loss(y_true, y_pred):
    return combo_loss(y_true, y_pred, alpha=0.5, beta=0.3)

# ---------- Metrics ----------
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()


# ---------------- Model Setup ----------------
IMG_SIZE = (256, 256)
MODEL_PATH = "models/seg_model_best.h5"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "combo_loss": combo_loss,
        "chosen_loss": chosen_loss,
        "dice_coef": dice_coef,
        "iou_coef": iou_coef,
        "soft_iou": soft_iou
    },
    compile=False
)


# ------------- LOGIN-------------
username = "guest"
            
def login_page():
    page_header("", "Login / Register")

    choice = st.radio("Choose an option:", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Register":
        if st.button("Register"):
            register_user(username, password)

    elif choice == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.page = "Detection"
                
                
# ------------- HOME TAB -------------
import streamlit.components.v1 as components

def show_home():
    # Read your HTML, CSS, and JS files
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    with open("templates/style.css", "r", encoding="utf-8") as f:
        css = f.read()
    with open("templates/script.js", "r", encoding="utf-8") as f:
        js = f.read()

    st.markdown("""
    <style>
    /* Remove all Streamlit padding and width limits */
    [data-testid="stAppViewContainer"], 
    [data-testid="stVerticalBlock"], 
    [data-testid="stVerticalBlock"] > div, 
    [data-testid="stHorizontalBlock"], 
    .block-container, 
    .main, 
    section[data-testid="stSidebar"] + section {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    /* Make the iframe stretch fully */
    iframe {
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

    # Combine them and render
    components.html(
        f"""
        <html>
        <head>
            <style>{css}</style>
        </head>
        <body>
            {html}
            <script>{js}</script>
        </body>
        </html>
        """,
        height=600,   # adjust to your HTML page size
        scrolling=True
    )

st.markdown("""
    <style>
    /* Remove top (and optionally all) padding from main container */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
    }

    /* Optional: remove margin around the body area */
    [data-testid="stAppViewContainer"] {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Optional: make header area transparent or remove it */
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
        height: 0rem;
    }
    
    [data-testid="stAppViewContainer"] > .main {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}

    </style>
""", unsafe_allow_html=True)
    



# ------------- DETECTION TAB -------------
def show_detection():
    page_header("", "Oil Spill Detection")
        
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.warning("⚠️ Login to save results and view history!")
        st.stop()
        
    if st.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

    username = st.session_state["username"]
    st.write(f"Welcome back, {username}!")

    uploaded_files = st.file_uploader("Upload satellite image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("🔍 Detect Oil Spill"):
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                img_resized = image.resize(IMG_SIZE)
                img_array = np.array(img_resized) / 255.0
                pred = model.predict(np.expand_dims(img_array, 0))[0, :, :, 0]
                mask = (pred > 0.5).astype(np.uint8) * 255

                # Overlay
                overlay = np.array(image.resize(IMG_SIZE)).copy()
                overlay[:, :, 0] = np.maximum(overlay[:, :, 0], mask)

                col1, col2, col3 = st.columns(3)
                col1.image(image, caption="Original Image", use_container_width=True)
                col2.image(mask, caption="Predicted Mask", use_container_width=True)
                col3.image(overlay, caption="Overlay", use_container_width=True)

                # --- Metrics section ---
                st.subheader("Model Metrics")
                colm = st.columns(5)
                colm[0].metric("Accuracy", "0.90")
                colm[1].metric("Precision", "0.90")
                colm[2].metric("Recall", "0.92")
                colm[3].metric("Dice", "0.90")
                colm[4].metric("IoU", "0.86")

                # ✅ Save to MongoDB
                orig_bytes = io.BytesIO()
                image.save(orig_bytes, format='PNG')
                mask_img = Image.fromarray(mask)
                mask_bytes = io.BytesIO()
                mask_img.save(mask_bytes, format='PNG')

                # Store in GridFS
                orig_id = fs.put(orig_bytes.getvalue(), filename=f"{uploaded_file.name}_original.png")
                mask_id = fs.put(mask_bytes.getvalue(), filename=f"{uploaded_file.name}_mask.png")

                # Get user info (or fallback to guest)
                user_id = st.session_state.get("username", "guest")

                # Store metadata
                record = {
                    "user": user_id,
                    "timestamp": datetime.datetime.utcnow(),
                    "filename": uploaded_file.name,
                    "original_image_id": orig_id,
                    "mask_image_id": mask_id
                }
                collection.insert_one(record)



                st.success("Detection results saved ✅")

                # ✅ Download button
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                st.download_button(
                    label="📥 Download Predicted Mask",
                    data=buf.getvalue(),
                    file_name=f"{uploaded_file.name.split('.')[0]}_mask.png",
                    mime="image/png",
                )

#-------HISTORY TAB -------------
def show_history():
    page_header("", "Detection History")


    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.warning("Please log in to view your history.")
        st.stop()

    if st.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

    username = st.session_state["username"]
    st.write(f"Welcome back, {username}!")

    results = get_user_history(username)
    if not results:
        st.info("No previous detections found.")
        return

    for r in results:
        with st.container():
            st.markdown(f"### {r['filename']}")
            st.caption(f"Detected on: {r['timestamp']}")

            col1, col2 = st.columns(2)
            col1.image(r["original"], caption="Original Image", use_container_width=True)
            col2.image(r["mask"], caption="Predicted Mask", use_container_width=True)

            # Metrics section
            if "metrics" in r and r["metrics"]:
                st.markdown("#### 📈 Model Metrics")
                metric_cols = st.columns(len(r["metrics"]))
                for (key, value), col in zip(r["metrics"].items(), metric_cols):
                    col.metric(key, value)

            st.markdown("---")


# ------------- MAIN NAVIGATION -------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar navigation
tab = st.sidebar.radio(
    "Navigation",
    ["Home", "Login", "Detection", "History"],
    index=0
)

if tab == "Home":
    show_home()
elif tab == "Login":
    login_page()
elif tab == "Detection":
    show_detection()
elif tab == "History":
    show_history()


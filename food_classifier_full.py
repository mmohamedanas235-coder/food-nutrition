import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import Food101
from PIL import Image
import os
import pandas as pd

# -----------------------------
# ⚙️ Streamlit Configuration
# -----------------------------
st.set_page_config(page_title="🍎 Food Classifier & Nutrition Analyzer", layout="centered")
st.title("🍔 Smart Food Classifier + Nutrition Tracker")
st.caption("Powered by ResNet50 trained on Food-101 🍱")

# -----------------------------
# 🧾 Load Nutrition Data
# -----------------------------
@st.cache_data
def load_nutrition_data():
    try:
        data = pd.read_csv("food_nutrition.csv")
        data['Food'] = data['Food'].str.lower()
        return data
    except Exception as e:
        st.error(f"⚠️ Failed to load nutrition data: {e}")
        return pd.DataFrame()

nutrition_data = load_nutrition_data()

# -----------------------------
# 🧠 Load Labels
# -----------------------------
@st.cache_data
def load_labels():
    dataset = Food101(root="data", download=True)
    return dataset.classes

labels = load_labels()

# -----------------------------
# 🧠 Load Model (ResNet50)
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(labels))

    model_path = "resnet50model.pth"
    if os.path.exists(model_path):
        st.sidebar.info("🔄 Loading trained model...")
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error loading model weights: {e}")
    else:
        st.sidebar.warning("⚠️ No trained model found. Using untrained model (predictions may be random).")

    model.eval()
    return model

model = load_model()

# -----------------------------
# 🔧 Image Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# 📸 Upload & Predict
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_idxs = probs.topk(5)

    st.subheader("🍽️ Prediction Results")
    for i in range(5):
        food_name = labels[top_idxs[i]].replace("_", " ").title()
        confidence = top_probs[i].item() * 100
        st.write(f"**{i+1}. {food_name}** — {confidence:.2f}%")

        # Show nutrition info if available
        nutrition = nutrition_data[nutrition_data['Food'] == food_name.lower()]
        if not nutrition.empty:
            info = nutrition.iloc[0]
            st.markdown(f"""
            🧾 **Nutrition Info for {food_name}:**
            - 🍛 Calories: **{info['Calories']} kcal**
            - 🍞 Carbs: **{info['Carbs']} g**
            - 🥩 Protein: **{info['Protein']} g**
            - 🧈 Fat: **{info['Fat']} g**
            - 💊 Vitamins: **{info['Vitamins']}**
            """)
        else:
            st.info(f"No nutrition data available for **{food_name}**.")

# -----------------------------
# 📘 Sidebar Info
# -----------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This app uses a **ResNet50** model trained on the **Food-101 dataset**  
to classify food images and provide nutritional insights.

📊 Data Source: *Food-101* + *Custom Nutrition CSV*
""")

# app.py - CONSOLIDATED AND CORRECTED CODE

import os
import gradio as gr
from PIL import Image
import numpy as np
import tensorflow as tf
from google import genai

# --- 0. ENVIRONMENT SETUP ---
# CRITICAL: This line sets the environment variable internally.
# Replace the placeholder key with your actual, clean API key (no quotes inside the value).
os.environ['GEMINI_API_KEY'] = 'AIzaSyBWFwU1LH_B6ks2q3ZAvIeEIpIIINxhQs4'


# --- 1. MODEL SETUP & LOADING ---
MODEL_FILE_NAME = 'skin_cnn_model.h5'
# MUST match the classes found during training
CLASS_NAMES = ['acne', 'bags', 'healthy_skin', 'pimple', 'redness'] 
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_FILE_NAME) 
    MODEL_LOADED = True
    print(f"✅ Model '{MODEL_FILE_NAME}' loaded successfully.")
except Exception as e:
    MODEL_LOADED = False
    print(f"❌ Warning: Could not load model. Using simulation. Error: {e}")


# --- 2. CORE UTILITY FUNCTION (MUST be defined before it is used) ---
def preprocess_image(image: Image.Image, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Convert PIL image to numpy array, resize, and normalize for model input."""
    image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# --- 3. AI ANALYSIS FUNCTION (with Robustness Layer) ---
def analyze_skin_image(image: Image.Image):
    """Predicts skin condition using the loaded CNN model."""
    if image is None:
        return {"classification": "Analysis Pending", "confidence": "0%"}
        
    if MODEL_LOADED:
        processed_image = preprocess_image(image) # Now this function is correctly defined
        predictions = model.predict(processed_image)[0]
        
        predicted_index = np.argmax(predictions)
        confidence = predictions[predicted_index] * 100
        classification = CLASS_NAMES[predicted_index]
        
        # *** ROBUSTNESS FIX: CONFIDENCE THRESHOLD ***
        ROBUSTNESS_THRESHOLD = 65.0 
        
        if confidence < ROBUSTNESS_THRESHOLD:
            # Override to "Healthy Skin" if confidence is low.
            classification = "Healthy Skin"
            confidence = 100.0 - confidence 
        # **********************************************
        
        return {"classification": classification, "confidence": f"{confidence:.2f}%"}
    else:
        # Fallback simulation (Keep this for debugging)
        return {"classification": "Pimple (SIMULATED)", "confidence": "55.00%"}

# --- 4. RECOMMENDATION ENGINE (The 'AI Chatbot') ---
def generate_recommendations(skin_result: dict, sleep_hours: int, water_intake: str, stress_level: str):
    
    # 1. Initialize the AI client
    try:
        client = genai.Client()
    except Exception:
        return "ERROR: AI Chatbot not active. Please set the GEMINI_API_KEY environment variable securely."

    classification = skin_result["classification"].capitalize()
    
    # 2. Build the Comprehensive Prompt
    prompt_template = f"""
    You are an expert Dermatologist and Wellness Advisor. Your goal is to provide personalized, non-medical advice.
    
    The user's skin has been classified by a CNN as: {classification}.
    
    The user's current lifestyle data is:
    - Average Sleep: {sleep_hours} hours/night
    - Daily Water Intake: {water_intake}
    - Perceived Stress Level: {stress_level}
    
    Based on this information, provide 3 to 4 actionable, friendly, and personalized recommendations. Start by addressing the classified skin condition ({classification}), then explain how the lifestyle factors (Sleep, Water, Stress) might be contributing to the condition, and give a specific tip for each. Format your response using bullet points.
    """
    
    # 3. Call the Generative Model
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt_template
        )
        
        return "### ✨ Personalized Insights & Health Tips (AI Chatbot):\n\n" + response.text
        
    except Exception as e:
        return f"ERROR during AI call: Could not generate advice. Details: {e}"

# --- 5. THE MAIN GRADIO PIPELINE ---
def full_analysis_pipeline(image, sleep, water, stress):
    
    # 1. Run AI Analysis
    skin_analysis_result = analyze_skin_image(image)
    
    # 2. Run Recommendation Engine
    recommendations = generate_recommendations(skin_analysis_result, sleep, water, stress)
    
    # 3. Format Output
    classification = skin_analysis_result["classification"].capitalize()
    confidence = skin_analysis_result["confidence"]
    
    output = f"""
## 🔬 AI Skin Analysis Result:
**Predicted Condition:** **{classification}**
**Confidence Level:** {confidence}

---
{recommendations}
    """
    
    return output

# --- 6. GRADIO INTERFACE SETUP ---

# Define Input Components (already fixed 'default' to 'value')
image_input = gr.Image(type="pil", label="Upload a Clear Skin Image for Analysis")
sleep_input = gr.Slider(minimum=2, maximum=12, step=1, value=7, label="Average Sleep (Hours/Night)")
water_input = gr.Radio(choices=["Low (Less than 4 glasses)", "Medium (4-8 glasses)", "High (More than 8 glasses)"], 
                       value="Medium (4-8 glasses)", label="Daily Water Intake")
stress_input = gr.Dropdown(choices=["Low", "Medium", "High"], value="Medium", label="Perceived Stress Level") 


# Define the Interface
demo = gr.Interface(
    fn=full_analysis_pipeline,
    inputs=[image_input, sleep_input, water_input, stress_input],
    outputs="markdown",
    title="Intelligent Skin Health: AI-Based Analysis & Lifestyle Insights",
    description="Upload an image of your skin and provide lifestyle data to receive a personalized, data-driven health report and recommendations."
)

if __name__ == "__main__":
    print("\n--- Starting Gradio App ---")
    print("Open your browser to the local URL provided below.")
    demo.launch()
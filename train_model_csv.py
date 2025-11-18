# train_model_csv.py

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# --- 1. Define Constants and Directories ---
# **!! IMPORTANT: SET YOUR PATHS HERE !!**
# This is the path to the FOLDER CONTAINING THE IMAGE SUBDIRECTORIES (acne, bags, redness).
# Based on your structure, this should be the 'files' folder.
BASE_DIR = 'files' 
# Path to your CSV file
CSV_FILE = 'skin_defects.csv' 

# Standard settings
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 15 

# train_model_csv.py

# --- 2. Load and Prepare Dataframe ---
try:
    df = pd.read_csv(CSV_FILE)
    
    # We will use only the 'front' image for simplicity and the 'type' for the label.
    
    # 1. Select the relevant columns
    df_train = df[['front', 'type']].copy()
    
    # 2. Rename columns to be compatible with flow_from_dataframe
    df_train.columns = ['filepath', 'label']
    
    # *** CRITICAL FIX ADDED HERE ***
    # Remove the leading forward slash ('/') from the filepath strings
    df_train['filepath'] = df_train['filepath'].str.lstrip('/')
    # ******************************

    # 3. Keras needs labels to be strings (categorical mode)
    df_train['label'] = df_train['label'].astype(str)
    
    # 4. Count the classes and define them
    NUM_CLASSES = len(df_train['label'].unique())
    CLASS_NAMES = sorted(df_train['label'].unique())
    # ... rest of the code ...
    print("\n--- Model Class Mapping ---")
    print(f"Found {NUM_CLASSES} classes: {CLASS_NAMES}")
    print("---------------------------\n")

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Could not load data or process CSV. Check paths: {e}")
    exit()


# --- 3. Data Augmentation and Generator Setup ---
datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    validation_split=0.2  # Use 20% of the data for validation
)

# Load Training Data from DataFrame
train_generator = datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=BASE_DIR,  # This is the base folder for the paths in the 'filepath' column
    x_col='filepath',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load Validation Data from DataFrame
validation_generator = datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=BASE_DIR,
    x_col='filepath',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False # Typically don't shuffle validation data
)

# --- 4. Build the CNN Model (Same structure as before) ---
model = Sequential([
    # Input Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    
    # Second Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Third Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25), 
    
    # Classification Head
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax') 
])

# --- 5. Compile and Train the Model ---
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print("--- Starting Model Training ---")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# --- 6. Save the Trained Model ---
MODEL_FILE_NAME = 'skin_cnn_model.h5'
model.save(MODEL_FILE_NAME)
print(f"\n--- SUCCESS: Model Saved as: {MODEL_FILE_NAME} ---")

# --- 7. Plotting Performance (Optional) ---
# ... (Plotting code from previous steps can be added here) ...
import os
import cv2
import numpy as np
import joblib
import time
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration (your paths)
IMAGE_DIR = "E:/Faruq/Business/DEEPRESENSE/Experimental/Simple land classification using Random Forest/binary classification/water body/image data"
LABEL_DIR = "E:/Faruq/Business/DEEPRESENSE/Experimental/Simple land classification using Random Forest/binary classification/water body/label data"
MODEL_PATH = "E:/Faruq/Business/DEEPRESENSE/Experimental/Simple land classification using Random Forest/binary classification/water body/XGBoost Algorithm/source code/trained_model.pkl"

start_time = time.time()

# Initialize lists
all_features = []
all_labels = []

# Get sorted files
image_files = sorted(os.listdir(IMAGE_DIR))
label_files = sorted(os.listdir(LABEL_DIR))

# Data loading and preprocessing
for img_file, lbl_file in zip(image_files, label_files):
    img_path = os.path.join(IMAGE_DIR, img_file)
    lbl_path = os.path.join(LABEL_DIR, lbl_file)

    img = cv2.imread(img_path)
    label = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)

    if img is None or label is None:
        print(f"⚠️ Skipping {img_file}/{lbl_file} - Failed to load")
        continue

    if img.shape[:2] != label.shape[:2]:
        print(f"🔧 Resizing {lbl_file} to match {img_file}")
        label = cv2.resize(label, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    all_features.append(img.reshape(-1, 3))
    all_labels.append(label.flatten())

# Combine data
features = np.concatenate(all_features)
labels = np.concatenate(all_labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42
)

# XGBoost training
print("🛠️ Training XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',  # For binary classification
    early_stopping_rounds=10  # Moved here from fit()
)

# Simple training with progress bar
with tqdm(total=100, desc="Training Progress") as pbar:
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    pbar.update(100)

# Save and evaluate
joblib.dump(model, MODEL_PATH)
print("✅ XGBoost model trained and saved.")

y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"🕒 Total runtime: {time.time() - start_time:.2f} seconds")

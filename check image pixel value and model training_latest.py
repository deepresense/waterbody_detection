import cv2
import numpy as np
import joblib
import time
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

start_time = time.time()

# Load RGB image and label image
img = cv2.imread("multicolor image.tif")
label_img = cv2.imread("multicolor label.tif", cv2.IMREAD_UNCHANGED)  # Assume single channel
label = label_img.astype(np.uint8)

assert img.shape[:2] == label.shape[:2], "RGB and label sizes differ!"

if img is not None and img.shape[2] == 3:
    print("✅ RGB image detected.")

    # Reshape features and labels
    features = img.reshape(-1, 3)
    labels = label.flatten()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print("🛠️ Training model...")

    # Simulated chunk training
    n_chunks = 10
    trees_per_chunk = 10
    model = RandomForestClassifier(n_estimators=trees_per_chunk, warm_start=True)

    pbar = tqdm(range(n_chunks), desc="Training Progress", unit="chunk")

    for i in pbar:
        model.n_estimators = (i + 1) * trees_per_chunk
        model.fit(X_train, y_train)

        elapsed = time.time() - start_time
        avg_time_per_chunk = elapsed / (i + 1)
        remaining_time = avg_time_per_chunk * (n_chunks - (i + 1))

        pbar.set_postfix({
            "Elapsed": f"{elapsed:.1f}s",
            "ETA": f"{remaining_time:.1f}s"
        })

    # Save the model
    joblib.dump(model, 'multicolor_model.pkl')
    print("✅ Model trained and saved.")

    # Evaluate model
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)

    print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

else:
    print("❌ Not an RGB image.")

end_time = time.time()
print(f"🕒 Total runtime: {end_time - start_time:.2f} seconds")

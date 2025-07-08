import cv2
import numpy as np
import joblib
import time
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

start_time = time.time()

# Load RGB image and label image
img = cv2.imread("multicolor image.tif")
label_img = cv2.imread("multicolor label.tif", cv2.IMREAD_UNCHANGED)  # Assume single channel
label = label_img.astype(np.uint8)
assert img.shape[:2] == label.shape[:2], "RGB and label sizes differ!"

if img is not None and img.shape[2] == 3:
    print("RGB image detected.")

    # Reshape features and labels
    features = img.reshape(-1, 3)
    labels = label.flatten()

    print("Training model...")

    # Simulated chunk training
    n_chunks = 10
    trees_per_chunk = 10
    total_estimators = n_chunks * trees_per_chunk

    model = RandomForestClassifier(n_estimators=trees_per_chunk, warm_start=True)

    pbar = tqdm(range(n_chunks), desc="Training Progress", unit="chunk")

    for i in pbar:
        model.n_estimators = (i + 1) * trees_per_chunk
        model.fit(features, labels)

        elapsed = time.time() - start_time
        avg_time_per_chunk = elapsed / (i + 1)
        remaining_time = avg_time_per_chunk * (n_chunks - (i + 1))

        pbar.set_postfix({
            "Elapsed": f"{elapsed:.1f}s",
            "ETA": f"{remaining_time:.1f}s"
        })

    joblib.dump(model, 'multicolor_model.pkl')
    print("✅ Model trained with RGB features.")
else:
    print("❌ Not an RGB image.")

end_time = time.time()
print(f"🕒 Total runtime: {end_time - start_time:.2f} seconds")

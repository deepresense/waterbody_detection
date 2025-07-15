import joblib
import cv2
import tkinter as tk
from tkinter import filedialog

def pick_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=filetypes)

def save_file(title, default_ext, filetypes):
    root = tk.Tk()
    root.withdraw()
    return filedialog.asksaveasfilename(title=title, defaultextension=default_ext, filetypes=filetypes)

# Select model, image, and output path
model_path = pick_file("Select Trained Model", [("Pickle files", "*.pkl")])
image_path = pick_file("Select Image File", [("Image files", "*.tif *.jpg *.png")])
output_path = save_file("Save Prediction Output", ".tif", [("TIFF files", "*.tif")])

# Load model and image
model = joblib.load(model_path)
new_img = cv2.imread(image_path)
new_features = new_img.reshape(-1, 3)

# Predict and save
predictions = model.predict(new_features)
pred_img = predictions.reshape(new_img.shape[:2])
mask = (pred_img * 255).astype('uint8')
cv2.imwrite(output_path, mask)
print(mask)
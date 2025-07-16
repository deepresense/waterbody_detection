# Import necessary libraries
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image

def select_folder(title="Select Folder"):
    """
    Opens a GUI dialog to select a folder.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(title=title)
    return folder_selected

def select_square_size():
    """
    Prompt user to enter square size for cropping.
    This can be replaced by GUI input if needed.
    For now, we use a simple input box in notebook.
    """
    while True:
        try:
            size = int(input("Enter desired square size for cropping (in pixels): "))
            if size > 0:
                return size
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

# Step 1: Select input folder
print("Step 1: Select input folder containing images...")
input_folder = select_folder("Select Input Folder")
if not input_folder:
    raise Exception("No input folder selected.")

# Step 2: Select output folder
print("Step 2: Select output folder to save cropped images...")
output_folder = select_folder("Select Output Folder")
if not output_folder:
    raise Exception("No output folder selected.")

# Step 3: Load all image files from input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
if not image_files:
    raise Exception("No image files found in input folder.")

# Step 4: Load all images and get their sizes
images = []
sizes = []

for img_file in image_files:
    img_path = os.path.join(input_folder, img_file)
    try:
        img = Image.open(img_path)
        img.load()  # Ensure image is fully loaded
        images.append((img_file, img))
        sizes.append(img.size)
    except Exception as e:
        print(f"Error loading {img_file}: {e}")

if not sizes:
    raise Exception("No valid images found to process.")

# Step 5: Find the smallest square dimension from all images
min_width = min(height for width, height in sizes)
min_height = min(height for width, height in sizes)
reference_size = min(min_width, min_height)

# Step 6: Ask user if they want to use a custom square size
use_custom_size = input("Do you want to define a custom square size for cropping? (y/n): ").strip().lower()
if use_custom_size == 'y':
    reference_size = select_square_size()

print(f"Reference square size for cropping: {reference_size}x{reference_size} pixels")

# Step 7: Crop all images to match the reference square size
for img_file, img in images:
    width, height = img.size
    
    # Calculate the cropping box
    left = (width - reference_size) // 2
    top = (height - reference_size) // 2
    right = left + reference_size
    bottom = top + reference_size

    # Crop the image to square
    cropped_img = img.crop((left, top, right, bottom))

    # Save the cropped image
    output_path = os.path.join(output_folder, f"cropped_{img_file}")
    cropped_img.save(output_path)
    print(f"Cropped and saved: {output_path}")

print("✅ All images have been cropped and saved successfully!")
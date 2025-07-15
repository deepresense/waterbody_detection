import numpy as np
import cv2
from tkinter import filedialog, Tk

# Open file dialog
Tk().withdraw()
path = filedialog.askopenfilename(title="Select Image")

# Load image without changing data
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

# Get unique values and counts
values, counts = np.unique(img, return_counts=True)
value_count = dict(zip(values, counts))

# Print result
print("Value Counts:", value_count)
if set(values).issubset({0, 1}):
    print("Image contains only 0 and 1.")
else:
    print("Image contains values other than 0 and 1.")

import rasterio
import numpy as np

# Open the raster
with rasterio.open('multicolor label.tif') as src:
    mask = src.read(1)  # Read the first band

# Count 0s and 1s
count_0 = np.sum(mask == 0)
count_1 = np.sum(mask == 1)

print(f"Count of 0s: {count_0}")
print(f"Count of 1s: {count_1}")

# Check for both 0 and 1
if count_0 > 0 and count_1 > 0:
    print("✅ The mask contains both 0 and 1.")
else:
    print("⚠️ The mask does NOT contain both 0 and 1.")

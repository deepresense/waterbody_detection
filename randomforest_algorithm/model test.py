import joblib
import cv2

model = joblib.load('trained_model.pkl')
new_img = cv2.imread("tile_57.tif")
new_features = new_img.reshape(-1, 3)
predictions = model.predict(new_features)
pred_img = predictions.reshape(new_img.shape[:2])
mask = (pred_img * 255).astype('uint8')
# # Apply colormap (e.g., red for water, black for non-water)
# colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
#
# cv2.imwrite("colored_mask.tif", colored)
cv2.imwrite("predicted_mask2.tif", pred_img.astype('uint8') * 255)

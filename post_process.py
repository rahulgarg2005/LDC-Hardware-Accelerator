import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the blurry/faint output image from your fused folder
# Replace with your exact image filename if different
image_path = r"result\BIPED2CLASSIC\fused\35028.png" 
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not find the image at {image_path}. Check your folder path!")
else:
    # 2. Apply a binary threshold
    # Any pixel value above 30 becomes 255 (white). Feel free to adjust '30'!
    threshold_value = 45 
    _, binarized_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    # 3. Save the crisp result
    output_path = r"result\BIPED2CLASSIC\fused\35028_crisp.png"
    cv2.imwrite(output_path, binarized_img)
    print(f"Success! Crisp edge map saved to: {output_path}")
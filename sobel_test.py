import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the original color image
# Using the same test image from your data folder
image_path = r"C:\Users\rahul\LDC\data\35028.jpg"
img_color = cv2.imread(image_path)

if img_color is None:
    print(f"Error: Could not find image at {image_path}")
else:
    # 2. Convert to Grayscale (Sobel operates on a single intensity channel)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 3. Apply Sobel Filter along X-axis (Detects vertical edges)
    # cv2.CV_64F handles negative gradients gracefully (e.g., transitions from white to black)
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)

    # 4. Apply Sobel Filter along Y-axis (Detects horizontal edges)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    # 5. Calculate Magnitude: G = sqrt(Gx^2 + Gy^2)
    sobel_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # 6. Normalize back to standard 8-bit image range (0 - 255)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))

    # 7. Save the raw Sobel output
    cv2.imwrite("result_sobel_raw.png", sobel_magnitude)

    # 8. Apply a Binary Threshold to see clean, sharp lines (Just like your post_process.py)
    threshold_value = 50
    _, sobel_thresholded = cv2.threshold(sobel_magnitude, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imwrite("result_sobel_binary.png", sobel_thresholded)

    # 9. Plot the input and outputs side-by-side using Matplotlib
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title("Original Input Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(sobel_magnitude, cmap='gray')
    plt.title("Raw Sobel Magnitude")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(sobel_thresholded, cmap='gray')
    plt.title(f"Thresholded Sobel (Val={threshold_value})")
    plt.axis("off")

    print("Success! Displaying plots. Close the window to finish.")
    plt.tight_layout()
    plt.show()
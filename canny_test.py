import cv2

# --- CONFIGURATION ---
INPUT_IMAGE_PATH  = r"C:\Users\rahul\LDC\data\35028.jpg"  # Put your original image name here
OUTPUT_IMAGE_PATH = 'canny_output.png' # What you want to name the saved file
# ---------------------

def main():
    # 1. Load the image in grayscale
    img_gray = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
        print(f"ERROR: Could not load {INPUT_IMAGE_PATH}. Check the path.")
        return

    # 2. Compute Canny edges (100 and 200 are standard thresholds)
    canny_edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)

    # 3. Save the result
    cv2.imwrite(OUTPUT_IMAGE_PATH, canny_edges)
    print(f"Success! Canny image saved as {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()
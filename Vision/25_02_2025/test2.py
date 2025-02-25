import cv2
import numpy as np

# 1. Load the image
image = cv2.imread('Img2.png')  # Replace with your image path

# 2. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Binarize the image (thresholding)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# 4. Detect edges (for grid lines)
edges = cv2.Canny(binary, 100, 200)

# 5. Use Hough Transform to detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# 6. Identify horizontal and vertical lines (simplified logic)
horizontal_lines = []
vertical_lines = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 10:  # Horizontal line (small vertical difference)
            horizontal_lines.append(line)
        elif abs(x1 - x2) < 10:  # Vertical line (small horizontal difference)
            vertical_lines.append(line)

# 7. Count squares by finding intersections or estimating grid size
# This step would require more complex logic to count the number of intersections
# or measure the distance between lines to determine rows and columns.

# For this image, it appears to be a 10x10 grid (based on visual estimation).
# You would need to refine this with actual line detection and intersection counting.

# Output: Print the number of squares
rows = len(horizontal_lines) - 1  # Number of spaces between horizontal lines
cols = len(vertical_lines) - 1   # Number of spaces between vertical lines
num_squares = rows * cols if rows > 0 and cols > 0 else 0

print(f"Number of squares in the grid: {num_squares}")
import numpy as np
import matplotlib.pyplot as plt
import cv2
#from scipy import ndimage
#from skimage import data

#import matplotlib.pyplot as plt
image_path = r"C:\Users\mahmo\Downloads\Test2.jpg"
RGB = cv2.imread(image_path)

plt.imshow(RGB)
plt.title('Colored Image')
plt.axis('off')  # Optional: Turn off axis labels
plt.show()
Gray = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)
plt.imshow(Gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')  # Optional: Turn off axis labels
plt.show()
# Define a few linear filters
Average_Filter = np.ones((7, 7)) / 49.0  # Averaging filter for blurring
Sharpen_Filter = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])-np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9.0
# Applying average filter to the gray image
filtered_image = cv2.filter2D(Gray, -1, Average_Filter)
plt.imshow(filtered_image, cmap='gray')
plt.title('Average filter')
plt.axis('off')  
plt.show()
# Applying average filter to colored image
filtered_RGB = cv2.filter2D(RGB, -1, Average_Filter)
plt.imshow(filtered_RGB)
plt.title('Average filter')
plt.axis('off')  
plt.show()
# Applying sharpenning filter to the image
Sharpened_image = cv2.filter2D(Gray, -1, Sharpen_Filter)
plt.imshow(Sharpened_image, cmap='gray')
plt.title('Sharpened image')
plt.axis('off')  
plt.show()
# Sobel filtering
sobel_x = cv2.Sobel(Gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(Gray, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the magnitude of the gradient
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Sobel filter ')
plt.axis('off')  
plt.show()
# Two-level thresholding 
low_threshold = 70
high_threshold = 120

edges = np.zeros_like(sobel_magnitude)
edges[sobel_magnitude > high_threshold] = 255
edges[(sobel_magnitude >= low_threshold) & (sobel_magnitude <= high_threshold)] = 50

plt.imshow(edges, cmap='gray')
plt.title('Thresholded Sobel Filtered Image ')
plt.axis('off')  
plt.show()
# Canny Filter
Canny_Filtered= cv2.Canny(Gray, 70, 120)
plt.imshow(edges, cmap='gray')
plt.title('Canny Filtered Image ')
plt.axis('off')  
plt.show()
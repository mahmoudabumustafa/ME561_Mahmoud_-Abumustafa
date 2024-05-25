import numpy as np
import matplotlib.pyplot as plt
import cv2

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
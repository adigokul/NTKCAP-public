
from PIL import Image
import numpy as np

# Open the image file
img = Image.open(r'E:\reID\0002_c1s1_000451_03.jpg')

# Convert image to a NumPy array
img_array1 = np.array(img)

# Display the shape of the array
print(img_array1.shape)

img = Image.open(r'E:\reID\8.jpg')

# Convert image to a NumPy array
img_array2 = np.array(img)

# Display the shape of the array
print(img_array2.shape)

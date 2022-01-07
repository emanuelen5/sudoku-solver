import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import hough


image_path = Path(__file__).parent.joinpath("example.jpg")
assert image_path.is_file()
img = cv2.imread(str(image_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
accum = hough.houghaccum(img)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img)
ax2.imshow(accum)
ax2.axis('equal')
plt.show()

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
import hough


image_path = Path(__file__).parent.joinpath("example.png")
assert image_path.is_file()
img = cv2.imread(str(image_path))
img_canny = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
accum = hough.houghaccum(img_canny)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1: Axes
ax2: Axes
ax3: Axes
ax1.imshow(img)
ax2.imshow(img_canny)
accum[accum == 0] = 1
ax3.imshow(accum, norm=LogNorm(vmin=1, vmax=1e6))
ax3.set_aspect(aspect="auto")
h, w = accum.shape
ax3.set_xlim(0, w)
ax3.set_ylim(0, h)
xs = np.arange(0, w, w/5)
ax3.set_xticks(xs, [str(x / 1) for x in xs])
ys = np.arange(0, h, h/10)
ax3.set_yticks(ys, [str(y - h/2) for y in ys])
ax3.set_xlabel("Angle [deg]")
ax3.set_ylabel("Distance [px]")
plt.show()

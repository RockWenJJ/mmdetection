import numpy as np
import matplotlib.pyplot as plt

input_img = '/home/wenjj/Documents/01_Projects/mmdetection/work_dirs/uformer_uie/results/data/real/image/3.png'
img = plt.imread(input_img)

img = 0.216 * img[:400, :400, 0] + 0.7152 * img[:400, :400, 1] + 0.0722 * img[:400, :400, 2]

fft2 = np.fft.fft2(img)

shift2center = np.fft.fftshift(fft2)
log_fft2 = np.log(1+np.abs(shift2center))

plt.imshow(log_fft2)
plt.show()
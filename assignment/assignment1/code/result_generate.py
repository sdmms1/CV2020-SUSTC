
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import vis_hybrid_image, load_image, save_image, im_range
from student_code import my_imfilter, create_hybrid_image

a, b = 'motorcycle', 'bicycle'

image1 = load_image('../data/' + a + '.bmp')
image2 = load_image('../data/' + b + '.bmp')
print(image1.shape)

cutoff_frequency = 7
filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4+1,
                               sigma=cutoff_frequency)
filter = np.dot(filter, filter.T)

low_frequencies, high_frequencies, hybrid_image = create_hybrid_image(image1, image2, filter)

cv2.imshow('low_frequencies', low_frequencies)
cv2.imshow('high_frequencies', high_frequencies)
cv2.imshow('hybrid_image', hybrid_image)

save_image('../results/low_frequencies_' + a + '.jpg', low_frequencies)
save_image('../results/high_frequencies_' + b + '.jpg', high_frequencies)
save_image('../results/' + a + '-' + b + '.jpg', hybrid_image)

cv2.waitKey()


import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *


def gaussian(arr, n, sigma):
    g = cv2.getGaussianKernel(n, sigma)
    return cv2.filter2D(arr, -1, g.dot(g.T))


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################
#     print(image.shape)
    # cv2.imshow("initial", image)
    if len(image.shape) != 2:
        if image.shape[2] == 1:
            image = image[..., 0]
        else:
            image = rgb2gray(image)
    # cv2.imshow("grey", image)

    g_image = gaussian(image, 7, 2)
    gx, gy = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # cv2.imshow("gx", gx);cv2.imshow("gy", gy)

    gx2, gy2, gxy = gx * gx, gy * gy, gx * gy
    # cv2.imshow("gx2", gx2);cv2.imshow("gy2", gy2);cv2.imshow("gxy", gxy)

    sx2, sy2, sxy = gaussian(gx2, 7, 2), gaussian(gy2, 7, 2), gaussian(gxy, 7, 2)
    # cv2.imshow("sx2", sx2); cv2.imshow("sy2", sy2); cv2.imshow("sxy", sxy)

    alpha = 0.06
    imageR = np.zeros(image.shape)
    imageR = sx2 * sy2 - sxy ** 2 - alpha * ((sx2 + sy2) ** 2)

    points = []
    for x in range(imageR.shape[0]):
        for y in range(imageR.shape[1]):
            points.append([y, x, imageR[x][y]])

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################

    points.sort(key=lambda x: x[2], reverse=True)
    points = points[0:int(len(points) * 0.01)]
#     print(len(points))

    d_order = []
    for i in range(len(points)):
        d = float("inf")
        for j in range(i):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dd = dx ** 2 + dy ** 2
            d = min(dd, d)
        d_order.append([points[i][0], points[i][1], d])

#     print(len(d_order))
    d_order.sort(key=lambda x: x[2], reverse=True)
    result = np.array(d_order)
    x = result[0:1500, 0]
    y = result[0:1500, 1]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, confidences, scales, orientations


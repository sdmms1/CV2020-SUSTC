import numpy as np


def my_imfilter(image, filter):
    """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    ### YOUR CODE HERE ###
    shape = image.shape

    h, w = filter.shape[0], filter.shape[1]
    fa, fb = h // 2, w // 2
    pad_matrix = np.zeros((shape[0] + 2 * fa, shape[1] + 2 * fb, shape[2]))
    for i in range(shape[2]):
        pad_matrix[:, :, i] = np.pad(image[:, :, i], ((fa, fa), (fb, fb)), 'constant')

    filtered_image = np.zeros(shape)

    for i in range(shape[2]):
        for x in range(shape[0]):
            for y in range(shape[1]):
                filtered_image[x][y][i] = np.sum(
                    np.multiply(pad_matrix[x: x + h, y: y + w, i], filter)
                )
    ### END OF STUDENT CODE ####
    ############################

    return filtered_image


def create_hybrid_image(image1, image2, filter):
    """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    ############################
    ### YOUR CODE HERE ###

    fp1, fp2 = my_imfilter(image1, filter), my_imfilter(image2, filter)
    fp1, fp2 = fp1 - fp1.min(), fp2 - fp2.min()
    fp1, fp2 = fp1 / fp1.max(), fp2 / fp2.max()
    low_frequencies, high_frequencies = fp1, image2 - fp2
    high_frequencies -= high_frequencies.min()
    high_frequencies /= high_frequencies.max()

    hybrid_image = low_frequencies + high_frequencies
    hybrid_image -= hybrid_image.min()
    hybrid_image /= hybrid_image.max()

    ### END OF STUDENT CODE ####
    ############################

    return low_frequencies, high_frequencies, hybrid_image

import numpy as np
from scipy import spatial
import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################
    
    n, m = features1.shape[0], features2.shape[0]
    match_points = []
    for i in range(n):
        first, second, im = float('inf'), float('inf'), -1
        for j in range(m):
            temp = np.linalg.norm(features1[i]-features2[j])
            if temp < first:
                second = first
                first = temp
                im = j
            elif temp < second:
                second = temp
        ratio = first / second
        match_points.append([i, im, ratio])

    match_points.sort(key=lambda x: x[2])
    matches = np.array(match_points)[0:100,0:2]
    confidences = np.array(match_points)[0:100,2]
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches.astype(int), confidences

if __name__ == '__main__':
    arr = [[1,2,3], [4,5,6]]
    print(np.array(arr)[0:1, 2])
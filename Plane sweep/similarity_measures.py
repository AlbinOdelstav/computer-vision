import numpy as np


def sad(x, y, images, window_size):
    window_half = int(window_size / 2)
    window_list = []

    for img in images:
        if not (0 < x - window_half < img.shape[0] and x + window_half + 1 < img.shape[0]) \
                or not (0 < y - window_half < img.shape[1] and y + window_half + 1 < img.shape[1]):
            return None, None
        window_list.append(img[x - window_half: x + window_half + 1, y - window_half: y + window_half + 1])

    color_sum = np.zeros(shape=(window_size, window_size, 3))
    for window in window_list:
        color_sum += window
    color_avg = color_sum / len(window_list)

    color_sum = 0
    for window in window_list:
        error = np.sum(np.absolute(window - color_avg))
        color_sum += error
    return color_sum, np.average(color_avg)


def ssd(x, y, images, window_size):
    window_half = int(window_size / 2)
    window_list = []

    for img in images:
        if not (0 < x - window_half < img.shape[0] and x + window_half + 1 < img.shape[0]) \
                or not (0 < y - window_half < img.shape[1] and y + window_half + 1 < img.shape[1]):
            return None, None
        window_list.append(img[x - window_half: x + window_half + 1, y - window_half: y + window_half + 1])

    color_sum = np.zeros(shape=(window_size, window_size, 3))
    for window in window_list:
        color_sum += window
    color_avg = color_sum / len(window_list)

    color_sum = 0
    for window in window_list:
        error = np.sum((window - color_avg) ** 2)
        color_sum += error
    return color_sum, np.average(color_avg)

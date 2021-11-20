import numpy

# Some example kernels


class Kernel:
    sharpen = numpy.array([[0.0, -1.0, 0.0],
                          [-1.0, 5.0, -1.0],
                          [0.0, -1.0, 0.0]])

    box_blur_5 = (1.0/25.0) * numpy.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0]])

    box_blur_3 = (1.0/9.0) * numpy.array([[1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0]])

    sobel = (1.0/8.0) * numpy.array([[-1.0, 0.0, 1.0],
                                     [-2.0, 0.0, 2.0],
                                     [-1.0, 0.0, 1.0]])

    emboss = numpy.array([[-1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0]])

    gaussian = (1.0/256.0) * numpy.array([[1.0, 4.0, 6.0, 4.0, 1.0],
                                          [4.0, 16.0, 24.0, 16.0, 4.0],
                                          [6.0, 24.0, 36.0, 24.0, 6.0],
                                          [4.0, 16.0, 24.0, 16.0, 4.0],
                                          [1.0, 4.0, 6.0, 4.0, 1.0]])

    box_blur_5_1d = (1.0/5.0) * numpy.array([[1.0, 1.0, 1.0, 1.0, 1.0]])

    box_blur_3_1d = (1.0/3.0) * numpy.array([[1.0, 1.0, 1.0]])

    sobel_1d_horizontal = (1.0/2.0) * numpy.array([[-1.0, 0.0, 1.0]])
    sobel_1d_vertical = (1.0/2.0) * numpy.array([[1.0],
                                                [2.0],
                                                [1.0]])

    gaussian_1d = 1.0 / 16.0 * numpy.array([[1.0, 4.0, 6.0, 4.0, 1.0]])

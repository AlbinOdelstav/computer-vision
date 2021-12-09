import math

import numpy
from PIL import Image, ImageDraw
import conv


def harris_response(x, y, ix2, iy2, ixy):
    h = numpy.array([[ix2[y, x], ixy[y, x]],
                    [ixy[y, x], iy2[y, x]]], dtype=float)
    k = 0.06
    trace = h[0, 0] + h[1, 1]
    return numpy.linalg.det(h) - (k * math.pow(trace, 2))


def draw_corners(pixels, threshold, output):
    draw = ImageDraw.Draw(output)
    for y in range(0, output.height):
        for x in range(0, output.width):
            if pixels[y, x] > threshold:
                draw.point([x, y], (255, 0, 0))


def get_derivatives(image):
    image_x = Image.new("L", image.size)
    image_x2 = Image.new("L", image.size)
    image_y = Image.new("L", image.size)
    image_y2 = Image.new("L", image.size)
    image_xy = Image.new("L", image.size)
    image_gray = image.convert("L")

    gaussian = (1.0/256.0) * numpy.array([[1.0, 4.0, 6.0, 4.0, 1.0],
                                          [4.0, 16.0, 24.0, 16.0, 4.0],
                                          [6.0, 24.0, 36.0, 24.0, 6.0],
                                          [4.0, 16.0, 24.0, 16.0, 4.0],
                                          [1.0, 4.0, 6.0, 4.0, 1.0]])

    sobel_horizontal = (1/8) * numpy.array([[1, 0, -1],
                                            [2, 0, -2],
                                            [1, 0, -1]])

    sobel_vertical = (1/8) * numpy.array([[1, 2, 1],
                                          [0, 0, 0],
                                          [-1, -2, -1]])

    conv.convolve(image_gray, image_x, sobel_horizontal)
    conv.convolve(image_gray, image_y, sobel_vertical)

    ix_pixels = numpy.asarray(image_x)
    iy_pixels = numpy.asarray(image_y)
    ix2_pixels = ix_pixels * ix_pixels
    iy2_pixels = iy_pixels * iy_pixels
    ixy_pixels = ix_pixels * iy_pixels

    conv.convolve(Image.fromarray(ix2_pixels), image_x2, gaussian)
    conv.convolve(Image.fromarray(iy2_pixels), image_y2, gaussian)
    conv.convolve(Image.fromarray(ixy_pixels), image_xy, gaussian)

    ix2 = numpy.asarray(image_x2)
    iy2 = numpy.asarray(image_y2)
    ixy = numpy.asarray(image_xy)

    return ix2, iy2, ixy


def find_corners(image):
    ix2, iy2, ixy = get_derivatives(image)
    harris_corners = numpy.zeros((image.height, image.width))
    for y in range(0, image.height):
        for x in range(0, image.width):
            harris_corners[y, x] = harris_response(x, y, ix2, iy2, ixy)
    return harris_corners


# oneliner?
def find_corners2(image):
    ix2, iy2, ixy = get_derivatives(image)
    return ix2 * iy2 - ixy * ixy - 0.06 * numpy.power((ix2 + iy2), 2)


def main():
    image = Image.open('../images/chess.png')
    harris_corners = find_corners(image)
    draw_corners(harris_corners, 40, image)
    image.show()
    image.save("chess_harris.png")


main()

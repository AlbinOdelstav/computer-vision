import numpy
from PIL import Image, ImageDraw
import conv


def harris_response(x, y, ix2, iy2, ixy, window):
    h = numpy.array([[0, 0],
                     [0, 0]])

    for dir_x in range(0, window.shape[0]):
        for dir_y in range(0, window.shape[0]):
            pos_x = x + dir_x
            pos_y = y + dir_y

            h += numpy.array([[ix2[pos_x, pos_y], ixy[pos_x, pos_y]],
                              [ixy[pos_x, pos_y], iy2[pos_x, pos_y]]])

    det = numpy.linalg.det(h)
    trace = h[0, 0] + h[1, 1]
    k = 0.06
    return det - (k * trace)


def draw_corners(pixels, threshold, output):
    draw = ImageDraw.Draw(output)
    for x in range(0, output.width):
        for y in range(0, output.height):
            if pixels[x, y] > threshold:
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

    sobel_horizontal = (1/8) * numpy.array([[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]])

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

    ix2_intermediate = Image.fromarray(ix2_pixels)
    iy2_intermediate = Image.fromarray(iy2_pixels)
    ixy_intermediate = Image.fromarray(ixy_pixels)

    conv.convolve(ix2_intermediate, image_x2, gaussian)
    conv.convolve(iy2_intermediate, image_y2, gaussian)
    conv.convolve(ixy_intermediate, image_xy, gaussian)

    ix2 = numpy.asarray(image_x2).transpose()
    iy2 = numpy.asarray(image_y2).transpose()
    ixy = numpy.asarray(image_xy).transpose()

    return ix2, iy2, ixy


def find_corners(image):
    ix2, iy2, ixy = get_derivatives(image)

    window = numpy.zeros((3, 3))
    window_half = int(window.shape[0] / 2)
    harris_corners = numpy.zeros((image.width, image.height))
    for x in range(window_half, image.width - window_half - 1):
        for y in range(window_half, image.height - window_half - 1):
            harris_corners[x, y] = harris_response(x, y, ix2, iy2, ixy, window)

    return harris_corners


# oneliner?
def find_corners2(image):
    ix2, iy2, ixy = get_derivatives(image)
    return ix2 * iy2 - ixy * ixy - 0.06 * numpy.power((ix2 + iy2), 2)


def main():
    image = Image.open('chess.png')
    harris_corners = find_corners(image)
    draw_corners(harris_corners, 0, image)
    image.show()
    image.save("chess_corners.png")


main()

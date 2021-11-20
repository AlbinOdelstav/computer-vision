import numpy
from kernel import Kernel
from PIL import Image, ImageDraw


# Makes sure the points in the kernel is mapped to points inside of the image
def get_kernel_bounds(x, y, kernel_width, kernel_height, image_width, image_height):
    x_min = max(0, kernel_width // 2 - x)
    x_max = min(kernel_width, image_width - x + 1)
    y_min = max(0, kernel_height // 2 - y)
    y_max = min(kernel_height, image_height - y + 1)
    return x_min, x_max, y_min, y_max


# Modifying a pixel by multiplying the weights of its neighbours with
# the corresponding points in the kernel
def calculate_pixel(kernel, pixels, image_width, image_height, x, y):
    kernel_height_half = int(kernel.shape[0]/2)
    kernel_width_half = int(kernel.shape[1]/2)
    x_min, x_max, y_min, y_max = get_kernel_bounds(x, y, kernel.shape[1], kernel.shape[0], image_width, image_height)
    rgb_sum = numpy.array([0, 0, 0])

    for kernel_x in range(x_min, x_max):
        for kernel_y in range(y_min, y_max):
            pos_x = x + kernel_x - kernel_width_half
            pos_y = y + kernel_y - kernel_height_half
            rgb_sum[0] += pixels[pos_x, pos_y][0] * kernel[kernel_y, kernel_x]
            rgb_sum[1] += pixels[pos_x, pos_y][1] * kernel[kernel_y, kernel_x]
            rgb_sum[2] += pixels[pos_x, pos_y][2] * kernel[kernel_y, kernel_x]
    return rgb_sum


# Goes through every pixel of an image and modifies them with a kernel
def convolve(image, output, kernel, weighted_average):
    kernel_size = (kernel.shape[0] * kernel.shape[1])
    draw = ImageDraw.Draw(output)
    pixels = image.load()

    for x in range(0, image.width):
        for y in range(0, image.height):
            rgb_sum = calculate_pixel(kernel, pixels, image.width, image.height, x, y)
            if weighted_average:
                rgb_sum = rgb_sum // kernel_size
            draw.point((x, y), (int(rgb_sum[0]), int(rgb_sum[1]), int(rgb_sum[2])))


# Performs two convolutions with a separable filter that has identical
# vertical and horizontal kernels
def convolve_seperable(image, output, kernel, weighted_average):
    tmp = Image.new("RGB", image.size)
    convolve(image, tmp, kernel, weighted_average)
    convolve(tmp, output, kernel.transpose(), weighted_average)


# Performs two convolutions with a separable filter that has different
# vertical and horizontal kernels
def convolve_seperable_2(image, output, kernel_1, kernel_2, weighted_average):
    tmp = Image.new("RGB", image.size)
    convolve(image, tmp, kernel_1, weighted_average)
    convolve(tmp, output, kernel_2, weighted_average)


def main():
    image = Image.open('frog1.jpg')
    output = Image.new("RGB", image.size)

    #convolve(image, output, Kernel.box_blur_5, True)
    convolve_seperable(image, output, Kernel.box_blur_5_1d, True)
    # convolve_seperable_2(image, output, Kernel.sobel_1d, Kernel.sobel_1d_2, True)

    output.show()
    output.save("box_blur.jpg")


main()


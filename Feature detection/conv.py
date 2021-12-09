from PIL import ImageDraw


def get_kernel_bounds(x, y, kernel_width, kernel_height, image_width, image_height):
    x_min = max(0, (kernel_width // 2) - x)
    x_max = min(kernel_width, (image_width - x) + 1)
    y_min = max(0, (kernel_height // 2) - y)
    y_max = min(kernel_height, (image_height - y) + 1)
    return x_min, x_max, y_min, y_max


def calculate_pixel(kernel, pixels, image_width, image_height, x, y):
    kernel_height_half = int(kernel.shape[0]/2)
    kernel_width_half = int(kernel.shape[1]/2)
    x_min, x_max, y_min, y_max = get_kernel_bounds(x, y, kernel.shape[1], kernel.shape[0], image_width, image_height)
    color_sum = 0

    for kernel_x in range(x_min, x_max):
        for kernel_y in range(y_min, y_max):
            pos_x = x + kernel_x - kernel_width_half
            pos_y = y + kernel_y - kernel_height_half
            color_sum += pixels[pos_x, pos_y] * kernel[kernel_y, kernel_x]
    return color_sum


def convolve(image, output, kernel):
    draw = ImageDraw.Draw(output)
    pixels = image.load()
    for y in range(0, image.height):
        for x in range(0, image.width):
            color_sum = calculate_pixel(kernel, pixels, image.width, image.height, x, y)
            draw.point((x, y), int(color_sum))

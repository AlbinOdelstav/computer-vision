import numpy


class Kernel:
    def __init__(self, n, size):
        self.n = n
        self.width = size
        self.height = size
        self.matrix = numpy.full((self.width, self.height), self.n)

    def calculate(self, img, new_img, x, y):
        img_width = img.shape[0]
        img_height = img.shape[1]
        kernel_width_half = int(self.width / 2)
        kernel_height_half = int(self.height / 2)

        for kernel_y in range(-kernel_height_half, kernel_height_half):
            for kernel_x in range(-kernel_width_half, kernel_width_half):
                kernel_offset_y = kernel_y + y
                kernel_offset_x = kernel_x + x
                if 0 <= kernel_offset_y < img_height and 0 <= kernel_offset_x < img_width:
                    new_img[x, y] = img[kernel_offset_x, kernel_offset_y] * self.matrix[kernel_x, kernel_y]
        return new_img

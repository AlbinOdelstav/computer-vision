import numpy
from kernel import Kernel
from PIL import Image


def create_filter(img, kernel):
    new_img = numpy.copy(img)
    img_width = img.shape[0]
    img_height = img.shape[1]
    for y in range(0, img_height):
        for x in range(0, img_width):
            kernel.calculate(img, new_img, x, y)
    return Image.fromarray(new_img, 'RGB')


def main():
    image = numpy.asarray(Image.open('frog1.jpg'))
    kernel = Kernel(20, 5)
    result = create_filter(image, kernel)
    result.save('frog1_got_filtered.jpg')
    result.show()

main()

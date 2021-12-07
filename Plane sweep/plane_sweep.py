import sys, getopt
import numpy as np
from PIL import Image, ImageDraw
from similarity_measures import ssd, sad


def matrix_horizontal_shift(matrix, steps):
    shift = matrix[0: matrix.shape[0] - steps: 1, 0: matrix.shape[1]: 1]
    matrix[steps: matrix.shape[0]: 1, 0: matrix.shape[1]: 1] = shift
    matrix[0: steps: 1, 0: matrix.shape[1]: 1] = 0


def draw_image(img, pixels):
    draw = ImageDraw.Draw(img)
    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            draw.point([x, y], (int(pixels[x, y][0]), int(pixels[x, y][1]), int(pixels[x, y][2])))
    return img


def plane_sweep(img_left, img_right, planes, depth_range, window_size, metric_function):
    img_left_pixels = np.asarray(img_left).transpose((1, 0, 2))
    img_right_pixels = np.asarray(img_right).transpose((1, 0, 2))

    vcam_colors = np.ones(shape=img_right_pixels.shape)
    vcam_score = np.full((img_right_pixels.shape[0], img_right_pixels.shape[1]), sys.maxsize)

    for plane_index in range(planes):
        depth = int((depth_range / planes) * plane_index)
        matrix_horizontal_shift(img_left_pixels, depth)

        print("Plane: {plane_index}\t Depth: {depth}".format(plane_index=plane_index + 1, depth=depth))
        for x in range(0, img_right_pixels.shape[0]):
            for y in range(0, img_right_pixels.shape[1]):
                color_score, color_avg = metric_function(x, y, [img_left_pixels, img_right_pixels], window_size)
                if color_score is None:
                    continue

                if color_score < vcam_score[x][y]:
                    vcam_score[x][y] = color_score
                    vcam_colors[x, y] = color_avg

    return draw_image(img_right, vcam_colors)


def main(argv):
    planes = 20
    window_size = 9
    depth_range = 40
    images_names = ["im0.jpg", "im1.jpg"]
    output_name = ""
    images = []
    metric_function_name = "ssd"
    metric_function = ssd

    try:
        opts, args = getopt.getopt(argv, "i:p:w:d:f:o:")
    except getopt.GetoptError:
        print('plane_sweep.py -i <"left_image right_image"> -p <planes> -d <depth range> -f <function> -o <output>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in "-i":
            images_names = str.split(arg)
        elif opt in "-o":
            output_name = arg
        elif opt in "-p":
            planes = int(arg)
        elif opt in "-d":
            depth_range = int(arg)
        elif opt in "-w":
            window_size = int(arg)
        elif opt in "-f":
            if str.lower(arg) == "ssd":
                metric_function_name = "ssd"
                metric_function = ssd
            elif str.lower(arg) == "sad":
                metric_function_name = "sad"
                metric_function = sad
            else:
                print("Invalid function \"" + arg + "\"")
                sys.exit(2)

    for image in images_names:
        images.append(Image.open(image))

    if output_name == "":
        output_name = str.split(images_names[0], ".")[0] + \
                      "_" + str(metric_function_name) + \
                      "_p" + str(planes) + "_d" + \
                      str(depth_range) + "_w" + \
                      str(window_size) + ".png"

    print("\n------------------------------------------")
    print("Starting plane sweep with configuration: ")
    print("Planes: ", planes)
    print("Window size: ", window_size)
    print("Depth range: ", depth_range)
    print("------------------------------------------\n")
    output = plane_sweep(images[0], images[1], planes, depth_range, window_size, metric_function)
    output.show()
    output.save(output_name)


if __name__ == "__main__":
    main(sys.argv[1:])

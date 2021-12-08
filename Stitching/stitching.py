import sys, getopt
import numpy
import numpy as np
import cv2
from matplotlib import pyplot as plt


def find_matches(des1, des2, threshold):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    return good_matches


def homography(kp1, kp2, matches):
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]

    A = numpy.zeros(shape=(len(list_kp1), 9))
    for i in range(0, len(list_kp1) // 2):
        A[2 * i, :] = numpy.array([list_kp1[i][0], list_kp1[i][1], 1, 0, 0, 0, -1 * list_kp2[i][0] * list_kp1[i][0],
                                   -1 * list_kp2[i][0] * list_kp1[i][1], -1 * list_kp2[i][0]])
        A[2 * i + 1, :] = numpy.array([0, 0, 0, list_kp1[i][0], list_kp1[i][1], 1, -1 * list_kp2[i][1] * list_kp1[i][0],
                                       -1 * list_kp2[i][1] * list_kp1[i][1], -1 * list_kp2[i][1]])

    u, s, v = np.linalg.svd(A)
    return v[v.shape[0] - 1: v.shape[0]: 1].reshape(3, 3)


def read_files(image_list):
    images_names = str.split(image_list)
    if len(images_names) is not 2:
        print("Two images must be provided, e.g.: -i \"image_left.png image_right.png\"")
        sys.exit(2)
    image_left = cv2.imread(images_names[0])
    image_right = cv2.imread(images_names[1])
    if image_left is None:
        print("Could not read file ", images_names[0])
        sys.exit(2)
    if image_right is None:
        print("Could not read file ", images_names[1])
        sys.exit(2)
    return image_left, image_right


def print_help():
    print('\nUsage: stitching.py -i "<image_left.png image_right"> -d -o <outputfile>"\n')
    print('Options\n')
    print('-h | --help\n\tShow this help message\n')
    print('-i | --input <"image_left image_right">\n\tSet input images, e.g.: -i "image_left.png image_right.png"\n')
    print('-d | --display \n\tDisplay the result image\n')
    print('-o | --output <outputfile>\n\tSave the result image\n')


def main(argv):
    image_left = None
    image_right = None
    output_name = None
    show_image = False
    threshold = 0.5

    try:
        opts, args = getopt.getopt(argv, "hdi:t:o:", ["help", "display", "input", "threshold", "output"])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()

        if opt in ("-d", "display"):
            show_image = True

        if opt in ("-i", "input"):
            image_left, image_right = read_files(arg)

        elif opt in ("-t", "threshold"):
            try:
                threshold = float(arg)
            except ValueError:
                print("Threshold must be a number")
                sys.exit(2)

        elif opt in ("-o", "output"):
            output_name = arg

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image_left, None)
    kp2, des2 = sift.detectAndCompute(image_right, None)

    matches = find_matches(des1, des2, threshold)
    H = homography(kp1, kp2, matches)

    width = image_left.shape[0] * 2
    height = image_left.shape[1]

    result = cv2.warpPerspective(image_left, H, (width, height))
    result[0:image_right.shape[0], 0:image_right.shape[1]] = image_right

    if output_name is not None:
        cv2.imwrite(output_name, result)

    if show_image:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])

import sys, getopt
import numpy
import numpy as np
import cv2
from matplotlib import pyplot as plt


def find_matches(des1, des2, threshold):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    print("Found {matches} matches".format(matches=len(matches)))
    good_matches = []
    while len(good_matches) < 4:
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)
        if len(good_matches) < 4:
            print("Found {good_matches} good matches, minimum 4 are needed. "
                  "Increasing threshold from {threshold} to {new_threshold}"
                  .format(good_matches=len(good_matches), threshold=threshold, new_threshold=threshold + 0.1))
            threshold += 0.01

    print("Accepted {good_matches} good matches".format(good_matches=len(good_matches)))
    return good_matches


def homography(kp1, kp2, matches):
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
    size = min(len(list_kp1), len(list_kp2))

    if size % 2 is not 0:
        size += 1

    A = numpy.zeros(shape=(size, 9))
    for i in range(0, size // 2):
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


def crop(image):
    image = np.delete(image,np.where(~image.any(axis=0))[0], axis=1)
    image = np.delete(image,np.where(~image.any(axis=1))[0], axis=0)
    return image


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

    size = (image_left.shape[1] * 2, image_left.shape[0])
    result = cv2.warpPerspective(image_left, H, size)
    result[0:image_right.shape[0], 0:image_right.shape[1]] = image_right

    result = crop(result)

    if output_name is not None:
        cv2.imwrite(output_name, result)

    if show_image:
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])

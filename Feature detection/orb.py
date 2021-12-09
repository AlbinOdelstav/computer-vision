import cv2

img = cv2.imread('../images/img_right.jpg', 0)

orb = cv2.ORB_create()
kp = orb.detect(img, None)
kp, des = orb.compute(img, kp)
img2 = cv2.drawKeypoints(img, kp, None, color=(0, 0, 255), flags=0)

cv2.imwrite('img_right_orb.png', img2)

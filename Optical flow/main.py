import numpy as np
import cv2

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

frame_1 = cv2.imread("./img_1.jpg")
frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
frame_2 = cv2.imread("./img_2.jpg")
frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(frame_1)
p0 = cv2.goodFeaturesToTrack(frame_1_gray, mask=None, **feature_params)
p1, st, err = cv2.calcOpticalFlowPyrLK(frame_1_gray, frame_2_gray, p0, None, **lk_params)

good_new = p1[st == 1]
good_old = p0[st == 1]

for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = np.array(np.round(new.ravel()), dtype=int)
    c, d = np.array(np.round(old.ravel()), dtype=int)
    mask = cv2.line(mask, (a, b), (c, d), [0, 0, 255], 2)
    frame_2 = cv2.circle(frame_2, (a, b), 5, [0, 0, 255], -1)

img = cv2.add(frame_2, mask)
cv2.imwrite("./output.png", img)

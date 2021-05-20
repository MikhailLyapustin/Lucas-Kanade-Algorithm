import cv2 as cv
import numpy as np
from scipy.signal import convolve2d
import time


def MyLucasKanade(previous_frame, current_frame, X, mask_size):
    previous_frame = cv.GaussianBlur(previous_frame, (71, 71), 0)
    current_frame = cv.GaussianBlur(current_frame, (71, 71), 0)

    previous_frame = np.array(previous_frame, dtype=float)
    current_frame = np.array(current_frame, dtype=float)

    n = mask_size // 2
    x, y = X[:, 1], X[:, 0]
    grads = []

    for i, j in zip(x, y):
        i, j = int(i), int(j)
        mask_previous = previous_frame[i - n:i + n + 1, j - n:j + n + 1]
        mask_current = current_frame[i - n:i + n + 1, j - n:j + n + 1]

        dI_dx = convolve2d(mask_previous, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode='valid')
        dI_dy = convolve2d(mask_previous, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), mode='valid')
        dI_dt = convolve2d(mask_previous, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), mode='valid') - convolve2d(mask_current, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), mode='valid')

        M = np.array([[np.sum(dI_dx ** 2), np.sum(dI_dx * dI_dy)],
                      [np.sum(dI_dx * dI_dy), np.sum(dI_dy ** 2)]])
        b = np.array([-np.sum(dI_dt * dI_dx), -np.sum(dI_dt * dI_dy)])

        try:
            grads.append(np.linalg.inv(M).dot(b))
        except:
            grads.append([0, 0])

    grads = np.array(grads)
    return X + grads, grads

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

#old_frame = cv.imread('picture1.png')
old_frame = cv.imread('kitti0.png')
height = old_frame.shape[0]
width = old_frame.shape[1]
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

n = 105
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
p0 = p0[:, 0, :]
p0 = np.array(list(filter(lambda arr: arr[0] > n//2 and arr[1] > n//2 and arr[0] < height - n//2 and arr[1] < width - n//2, p0)))
p0 = np.array([[i] for i in p0])

#frame = cv.imread('picture2.png')
frame = cv.imread('kitti1.png')
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# calculate optical flow
time1 = time.time()
p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
timeCV = time.time() - time1

time2 = time.time()
p2, deltas_2 = MyLucasKanade(old_gray, frame_gray, p0[:, 0, :], n)
timeMy = time.time() - time2

# draw the tracks
mask = np.zeros_like(old_frame)
mask1 = np.zeros_like(old_frame)
frame1 = cv.imread('kitti1.png')
for new, new1, old in zip(p1, p2, p0):
    a, b = new.ravel()
    a1, b1 = new1.ravel()
    c, d = old.ravel()
    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), [0, 0, 255], 2)
    frame = cv.circle(frame, (int(c), int(d)), 3, [0, 0, 255], -1)
    mask1 = cv.line(mask1, (int(a1), int(b1)), (int(c), int(d)), [0, 0, 255], 2)
    frame1 = cv.circle(frame1, (int(c), int(d)), 3, [0, 0, 255], -1)
    old_frame = cv.circle(old_frame, (int(c), int(d)), 3, [0, 0, 255], -1)
img = cv.add(frame, mask)
img1 = cv.add(frame, mask1)

p1, p0 = p1[:, 0, :], p0[:, 0, :]
sq = []
for i, j, k in zip(p1, p2, p0):
    a = i - k
    b = j - k
    print('-----------------------------\n', a, '\n', b)
    sq.append((a - b) ** 2)
print(f'\nMSE: {np.sum(sq) / len(sq)}')
print(f'Time LK OpenCV: {"%.6f" % timeCV}\nTime MyLK: {"%.6f" %timeMy}')

cv.imshow('LK OpenCV',img)
cv.imshow('My LK',img1)
cv.imshow('Old Frame', old_frame)

#cv.imwrite('LK_OpenCV.png', img)
#cv.imwrite('My_LK.png', img1)
#cv.imwrite('Old_frame.png', old_frame)

cv.waitKey()

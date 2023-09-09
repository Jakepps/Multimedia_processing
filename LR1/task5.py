import cv2

img1 = cv2.imread(r'.\LR1\img2.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r'.\LR1\img2.png',cv2.IMREAD_REDUCED_COLOR_8)

cv2.namedWindow('gosling', cv2.WINDOW_NORMAL)
cv2.namedWindow('ghost', cv2.WINDOW_NORMAL)

cv2.imshow('gosling',img1)

hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
cv2.imshow('ghost', hsv)


cv2.waitKey(0)
cv2.destroyAllWindows()


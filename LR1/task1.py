import cv2

img1 = cv2.imread(r'.\LR1\img1.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r'.\LR1\img2.png',cv2.IMREAD_REDUCED_COLOR_8)
img3 = cv2.imread(r'.\LR1\img3.bmp',cv2.IMREAD_ANYDEPTH)
cv2.namedWindow('gosling', cv2.WINDOW_NORMAL)
cv2.namedWindow('ghost', cv2.WINDOW_NORMAL)
cv2.namedWindow('nature', cv2.WINDOW_NORMAL)
cv2.imshow('gosling',img1)
cv2.imshow('ghost', img2)
cv2.imshow('nature', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()


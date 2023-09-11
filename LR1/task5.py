import cv2

img1 = cv2.imread(r'.\LR1\source\img2.png')
img2 = cv2.imread(r'.\LR1\source\img2.png')

cv2.namedWindow('ghost', cv2.WINDOW_NORMAL)
cv2.namedWindow('ghost_hsv', cv2.WINDOW_NORMAL)

cv2.imshow('ghost',img1)

hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
cv2.imshow('ghost_hsv', hsv)


cv2.waitKey(0)
cv2.destroyAllWindows()


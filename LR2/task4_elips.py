import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 220])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    moments = cv2.moments(mask)
    area = moments['m00']

    if area > 0:
        width = height = int(np.sqrt(area))

        c_x = int(moments["m10"] / moments["m00"])
        c_y = int(moments["m01"] / moments["m00"])

        cv2.ellipse(frame,
            (c_x, c_y),
            (width // 16, height // 16),
            0,  # угол поворота эллипса
            0, 360, # начальный и конечный угол дуги
            (0, 0, 0), 2)

    cv2.imshow('Ellipse_frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

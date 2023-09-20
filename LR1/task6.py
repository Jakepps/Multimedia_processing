import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pointsUp = np.array([[280, 80], [360, 80], [320, 300]])
    pointsLeft = np.array([[180, 200], [180, 280], [360, 240]])
    pointsDown = np.array([[280, 420], [360, 420], [320, 200]])
    pointsRight = np.array([[460, 200], [460, 280], [260, 240]])

    result_frame = cv2.addWeighted(frame, 1, frame, 0.5, 0)

    cv2.fillPoly(result_frame, pts=[pointsUp], color=(255, 0, 0))
    cv2.fillPoly(result_frame, pts=[pointsLeft], color=(255, 0, 0))
    cv2.fillPoly(result_frame, pts=[pointsDown], color=(255, 0,0))
    cv2.fillPoly(result_frame, pts=[pointsRight], color=(255, 0, 0))

    result_frame = cv2.addWeighted(frame, 1, result_frame, 0.5, 0)

    cv2.imshow("Red Cross", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

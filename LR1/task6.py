import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    cross_image = np.zeros((height, width, 3), dtype=np.uint8)

    vertical_line_width = 60
    vertical_line_height = 300
    cv2.rectangle(cross_image,
                  (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2),
                  (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2),
                  (0, 0, 255), 2)

    horizontal_line_width = 250
    horizontal_line_height = 55
    cv2.rectangle(cross_image,
                  (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2),
                  (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2),
                  (0, 0, 255), 2)

    result_frame = cv2.addWeighted(frame, 1, cross_image, 0.5, 0)

    cv2.imshow("Red Cross", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

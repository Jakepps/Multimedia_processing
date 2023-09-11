import cv2

cap = cv2.VideoCapture(r'.\LR1\source\video.mp4', cv2.CAP_ANY)

new_width = 640
new_height = 480

while True:
    ret, frame = cap.read()

    if not ret:
        exit()

    frame = cv2.resize(frame, (new_width, new_height))

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Video', gray_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        exit()

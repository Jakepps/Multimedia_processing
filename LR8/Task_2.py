import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

# загрузка модели YOLOv3 для обнаружения лиц
net = cv2.dnn.readNetFromDarknet('resources_for_yolo/yolov3-face.cfg','resources_for_yolo/yolov3-wider_16000.weights')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
res = cv2.VideoWriter('result/yolo_output_1.avi', fourcc, 20.0, (640, 480))

frame_count = 0
prev_frame_time = 0

start_time = time.time()

while True:
    ret, frame = cap.read()

    # получение списка обнаруженных лиц
    frame_count += 1

    current_time = time.time()
    time_diff = current_time - prev_frame_time

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (415, 415), [0, 0, 0], True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # отрисовка прямоугольников вокруг лиц
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.98 and classId == 0:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)

    res.write(frame)

    if time_diff > 1:
        fps = frame_count / time_diff
        print(f"Частота потери изображения: {1 / ((current_time - prev_frame_time) / frame_count):.0f} кадр(-a)(-ов)/секунду")
        prev_frame_time = current_time
        frame_count = 0

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

end_time = time.time()

print(f"Время работы метода: {end_time - start_time:.5f} секунд")
print(f"Скорость обработки: {cap.get(cv2.CAP_PROP_FPS):.0f} fps")

cap.release()
res.release()
cv2.destroyAllWindows()

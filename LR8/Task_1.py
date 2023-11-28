import cv2
import time

cap = cv2.VideoCapture(0)

# загрузка каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier('sources_for_haarscade/haarcascade_frontalface_default.xml')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result/haarscade_output.avi', fourcc, 20.0, (640, 480))

# задание переменных для подсчета частоты потери изображения
frame_count = 0
prev_frame_time = 0

start_time = time.time()

while True:
    ret, frame = cap.read()

    if ret:
        frame_count += 1

        current_time = time.time()
        time_diff = current_time - prev_frame_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # обнаружение лиц на кадре
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(frame)

        if time_diff > 1:
            fps = frame_count / time_diff
            print(
                f"Частота потери изображения: {1 / ((current_time - prev_frame_time) / frame_count):.0f} кадр(-a)(-ов)/секунду")
            prev_frame_time = current_time
            frame_count = 0

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

end_time = time.time()


print(f"Время работы метода: {end_time - start_time:.5f} секунд")
print(f"Скорость обработки: {cap.get(cv2.CAP_PROP_FPS):.0f} fps")

cap.release()
out.release()
cv2.destroyAllWindows()

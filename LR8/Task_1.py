import cv2

cap = cv2.VideoCapture(0)

# загрузка каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier('sources_for_haarscade/haarcascade_frontalface_default.xml')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result/haarscade_output_1.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # обнаружение лиц на кадре
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()

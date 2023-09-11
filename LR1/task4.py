import cv2

def readIPWriteTOFile():
    video = cv2.VideoCapture(r'.\LR1\source\video.mp4', cv2.CAP_ANY)
    ok, vid = video.read()

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("./LR1/Output/output3.mp4", fourcc, 25, (w, h))

    while (True):
        ok, vid = video.read()

        cv2.imshow('Video', vid)
        video_writer.write(vid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()

readIPWriteTOFile()

import os
import cv2 as cv

# init tracker
tracker = cv.TrackerCSRT_create()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Video could not be opened')


# relative path to exact path
def resource_path(relative):
    return os.path.join(
        os.environ.get(
            "_MEIPASS2",
            os.path.abspath(".")
        ),
        relative
    )


detected = False
success = False
while True:
    _, frame = cap.read()
    frame = cv.flip(frame, 1)
    start = cv.getTickCount()

    if not detected:
        if frame is not None:
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            stop_haarcascade = cv.CascadeClassifier(resource_path('stop_data.xml'))
            roi = stop_haarcascade.detectMultiScale(frame_gray, 1.3, 5)

            # roi = cv.selectROI("tracker", frame)
        else:
            raise ValueError("Error: Frame is empty or invalid")

        amount_found = len(roi)

        if amount_found != 0:
            roi = tuple(roi.flatten())
            try:
                tracker.init(frame, roi)
                detected = True
            except Exception as e:
                print(e)

    elif detected:
        success, roi = tracker.update(frame)
        roi = tuple(int(i) for i in roi)
        if success:
            cv.rectangle(frame, roi, (0, 255, 0), 2)
        else:
            # detected = False
            pass

    fps = cv.getTickFrequency() / (cv.getTickCount() - start)

    # show pfs
    cv.putText(frame, f"FPS: {round(fps)}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.putText(frame, "Detected" if success else "No Detection", (10, 50),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if success else (255, 0, 0))

    cv.imshow("tracker", frame)

    key = cv.waitKey(1)
    if key == ord('q') or key == 27:
        break

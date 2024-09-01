import os
import cv2 as cv
import numpy as np
from ultralytics import YOLO

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Video could not be opened')


def reset():
    global detected, gotBox, detByCascade, mouseDown, mouseUp, drawBox,\
        start_x, start_y, track_initialized, rois, trackers
    detected = False
    gotBox = False
    detByCascade = False
    mouseDown = False
    mouseUp = False
    drawBox = np.zeros(4)
    start_x = 0
    start_y = 0
    track_initialized = False
    rois = ()
    trackers = []


reset()


# relative path to exact path
# When developer want to make this program an executable
# relative paths would not work, this code returns exact paths of
# relative paths for every program
def resource_path(relative):
    return os.path.join(
        os.environ.get(
            "_MEIPASS2",
            os.path.abspath(".")
        ),
        relative
    )


def on_mouse(event, x, y, flags, param):
    global mouseDown, mouseUp, start_x, start_y, roi, track_initialized, drawBox
    if event == cv.EVENT_LBUTTONDOWN:
        mouseDown = True
        mouseUp = False
        start_x = x
        start_y = y
        drawBox[0] = start_x
        drawBox[1] = start_y
    elif mouseDown and event == cv.EVENT_MOUSEMOVE:
        if x > start_x:
            drawBox[2] = x - start_x
        else:
            drawBox[2] = start_x - x
            drawBox[0] = x
        if y > start_y:
            drawBox[3] = y - start_y
        else:
            drawBox[3] = start_y - y
            drawBox[1] = y
    elif event == cv.EVENT_LBUTTONUP:
        mouseDown = False
        mouseUp = True
        track_initialized = True


model = YOLO("best.pt")

while True:
    _, frame = cap.read()
    frame = cv.flip(frame, 1)
    frame_process = frame.copy()
    start = cv.getTickCount()

    if mouseDown:
        drawBox = [int(i) for i in drawBox]
        pt1 = (drawBox[0], drawBox[1])
        pt2 = (drawBox[0] + drawBox[2], drawBox[1] + drawBox[3])
        cv.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

    # This should be run in every bunch of second
    if not detected:
        if frame is not None:
            # frame_gray = cv.cvtColor(frame_process, cv.COLOR_BGR2GRAY)
            # frame_rgb = cv.cvtColor(frame_process, cv.COLOR_BGR2RGB)
            # stop_haarcascade = cv.CascadeClassifier(resource_path('stop_data.xml'))
            # detectedBox = stop_haarcascade.detectMultiScale(frame_gray, 1.3, 5)
            results = model.predict(source=frame, save=False)
            print(f"class: {results.boxs.cls}\nconf: {results.boxs.cls}") # broken
            detectedBox = results[0].boxes.xyxy.cpu().numpy().astype(int)
            # if haar cascade detect any thing
            if len(detectedBox) > 0:
                detectedBox[0][2] = detectedBox[0][2] - detectedBox[0][0]
                detectedBox[0][3] = detectedBox[0][3] - detectedBox[0][1]
                roi = detectedBox
                track_initialized = True
                detected = True
                detByCascade = True
        else:
            raise ValueError("Error: Frame is empty or invalid")

    if track_initialized:
        if detByCascade:
            roi = tuple(roi.flatten())
            rois += tuple(roi[i:i + 4] for i in range(0, len(roi), 4))
            gotBox = True
            detByCascade = False
        else:
            roi = tuple(drawBox)
            rois = rois + (roi,)
            gotBox = True
        try:
            for i in range(len(rois)):
                roi = rois[i]
                tracker = cv.TrackerCSRT_create()
                trackers.append(tracker)
                trackers[i].init(frame_process, roi)
        except Exception as e:
            print(e)
        track_initialized = False

    if gotBox:
        for i in range(len(rois)):
            roi = rois[i]
            _, roi = trackers[i].update(frame_process)
            roi = tuple(int(i) for i in roi)
            x, y, w, h = roi
            # If the detected image get smaller its label also should get smaller
            # that is why we give font scale w/100. Mean w is 100 and font scale is 1
            # if w become like 33 the font scale will be 0.3
            cv.putText(frame, f"object {i}", (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, w/125, (0, 200, 0), 2)
            cv.rectangle(frame, roi, (0, 255, 0), 2)

    fps = cv.getTickFrequency() / (cv.getTickCount() - start)

    # show pfs
    cv.putText(frame, f"FPS: {round(fps)}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.putText(frame, "Detected" if detected else "No Detection", (10, 50),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if detected else (255, 0, 0))

    cv.imshow("tracker", frame)
    cv.setMouseCallback("tracker", on_mouse)

    key = cv.waitKey(1)
    if key == 27:
        break
    elif key == ord('q'):
        reset()

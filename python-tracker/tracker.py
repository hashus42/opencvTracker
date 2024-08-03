import cv2 as cv

# init tracker
tracker = cv.TrackerCSRT_create()

cap = cv.VideoCapture(0)
_, frame = cap.read()
frame = cv.flip(frame, 1)

# program stops here and wit for to input as drawing rectangle
roi = cv.selectROI("tracker", frame)
tracker.init(frame, roi)

while True:
    start = cv.getTickCount()

    _, frame = cap.read()
    frame = cv.flip(frame, 1)

    success, roi = tracker.update(frame)

    if success:
        cv.rectangle(frame, roi, (0, 255, 0), 2)

    fps = cv.getTickFrequency() / (cv.getTickCount() - start)

    # show pfs
    cv.putText(frame, f"FPS: {round(fps)}", [10, 30],
               cv.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255])

    cv.imshow("tracker", frame)

    key = cv.waitKey(1)
    if key == ord('q') or key == 27:
        break


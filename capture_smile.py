import cv2 as cv
import timeit
import os
import time

## Setup global variables
cap = cv.VideoCapture(0)
face_detection = cv.CascadeClassifier("face_detect_src.xml")
smile_detection = cv.CascadeClassifier("smile_detect_src.xml")
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
smile_images_path = os.path.join(BASE_PATH, "smile_images")
start = timeit.default_timer()
smile_counter_time = timeit.default_timer()
smile_counter = 0
counter = 0

# Making while loop for creating video
while(True):
    ret, frame = cap.read()
    frame_copy = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_rect = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if timeit.default_timer() - smile_counter_time > 1:
        smile_counter_time = timeit.default_timer()
        smile_counter = 0
    nb_smile = 0
    face_rect_len = 0
    for (x, y, w, h) in face_rect:
        face_rect_len += 1
        cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi = gray[y:y+w, x:x+w]
        roi_resize = cv.resize(roi, (300,300), interpolation=cv.INTER_AREA)
        smile_rect = smile_detection.detectMultiScale(roi_resize, scaleFactor=5.0, minNeighbors=15)
        flag = False
        for (xs, ys, ws, hs) in smile_rect:
            if not flag:
                nb_smile += 1
                flag = True
            #cv.rectangle(frame, (x+xs,y+ys), (x+xs+ws, y+ys+hs), (0,255,0), 2)
            smile_counter += 1
    if nb_smile == face_rect_len and  timeit.default_timer() - start > 5.0 and smile_counter > 5*face_rect_len:
        start = timeit.default_timer()
        smile_counter = 0
        while True:
            if not os.path.exists(os.path.join(smile_images_path,"smile_faces_anime{}.png".format(counter))):
                frame_copy_flip = cv.flip(frame_copy, 1)
                frame_copy_gray = cv.cvtColor(frame_copy_flip, cv.COLOR_BGR2GRAY)
                frame_copy_gray_blur = cv.GaussianBlur(frame_copy_gray, (31,31), 3.5, borderType=cv.BORDER_DEFAULT)
                frame_copy_adapt = cv.adaptiveThreshold(frame_copy_gray_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 1)
                frame_copy_blur = cv.blur(frame_copy_flip, (51,51))
                frame_copy_anime = cv.bitwise_and(frame_copy_blur, frame_copy_blur, mask = frame_copy_adapt)
                cv.imwrite(os.path.join(smile_images_path,"smile_faces_anime{}.png".format(counter)), frame_copy_anime)
                cv.imwrite(os.path.join(smile_images_path, "smile_faces_normal{}.png".format(counter)),frame_copy_flip)
                time.sleep(2.0)
                counter += 1
                break
            else:
                counter += 1


    flip = cv.flip(frame, 1)
    cv.imshow("Face, smile, face profile detection", flip)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()



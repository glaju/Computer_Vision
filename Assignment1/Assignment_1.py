import sys
import numpy as np
import cv2
import imutils

def toGrayscale(frames):
    for i in range(len(frames)):
        frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
    return frames

def toGaussianBlur(frames):
    for i in range(len(frames)):
        frames[i] = cv2.GaussianBlur(frames[i], (2*int(i/5)+1,2*int(i/5)+1), 0)
    return frames

def toBilateral(frames):
    k = len(frames)
    for i in range(len(frames)):
        frames[i] = cv2.bilateralFilter(frames[i], 9, 75, 75)
        if i>k/3:
            frames[i] = cv2.bilateralFilter(frames[i], 9, 75, 75)
            if i> (2*k/3):
                frames[i] = cv2.bilateralFilter(frames[i], 9, 75, 75)

    return frames

def filterRGB(frames):

    lower = (50, 50, 1)
    upper = (255, 255, 60)

    lower2 = (80, 100, 60)
    upper2 = (230, 230, 120)

    for i in range(len(frames)):
        mask = cv2.inRange(frames[i], lower, upper) + cv2.inRange(frames[i], lower2, upper2)
        frames[i] = cv2.bitwise_and(frames[i], frames[i], mask=mask)


    return frames

def filterHSV(frames):
    lower = (70, 0, 50)
    upper = (130, 255,255)
    for i in range(len(frames)):
        frame_hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_hsv, lower, upper)
        frames[i] = cv2.bitwise_and(frames[i], frames[i], mask=mask)
    return frames

def filterLAB(frames):
    lower = (40, 0, 0)
    upper = (255, 150, 120)
    for i in range(len(frames)):
        frame_hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(frame_hsv, lower, upper)
        frames[i] = cv2.bitwise_and(frames[i], frames[i], mask=mask)
    return frames

def filterHSV_morph(frames):
    lower = (70, 0, 50)
    upper = (130, 255,255)


    kernel = np.ones((5, 5), np.uint8)

    for i in range(len(frames)):
        frame_hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_hsv, lower, upper)
        frames[i] = cv2.bitwise_and(frames[i], frames[i], mask=mask)
        morph_modified = cv2.erode(mask, kernel, iterations=1)
        morph_modified = cv2.dilate(morph_modified, kernel, iterations=3)
        diff = cv2.subtract(morph_modified, mask)

        frames[i][np.where(diff != 0)] = [0, 0, 255]
        #frames[i] = diff

    return frames


def invisible_tower(frames, background):
    tmp = background[:,60:90,:]
    for i in range(10,640,30):
        background[:,i:i+30,:] = tmp
    background = cv2.GaussianBlur(background, (5, 5), 0)

    lower = (70, 0, 50)
    upper = (130, 255, 255)

    kernel = np.ones((5, 5), np.uint8)

    for i in range(len(frames)):
        frame_hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_hsv, lower, upper)

        morph_modified = cv2.dilate(mask, kernel, iterations=30)
        morph_modified = cv2.erode(morph_modified, kernel, iterations=25)
        frames[i][np.where(morph_modified == 255)] = background[np.where(morph_modified == 255)]

    return frames


def change_color(frames):
    lower = (50, 0, 50)
    upper = (150, 255, 255)
    for i in range(len(frames)):
        frame_hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_hsv, lower, upper)

        a = [frames[i][np.where(mask)][:, 1],
             frames[i][np.where(mask)][:, 0],
             frames[i][np.where(mask)][:, 2]]
        frames[i][np.where(mask)] = np.array(a).T.tolist()

    return frames


def detect_faces(frames):
    for i in range(len(frames)):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frames[i], (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frames[i][y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return frames


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    cap = cv2.VideoCapture('./video.MPEG')
    fps = cap.get(cv2.CAP_PROP_FPS)



    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            frames.append(frame)
        else:
            break


    cap.release()
    cv2.destroyAllWindows()
    #0-5
    frames[:150] = toGrayscale(frames[:150])
    #5-10
    frames[150:300] = toGaussianBlur(frames[150:300])
    #10-15
    frames[300:450] = toBilateral(frames[300:450])

    #20-25
    frames[600:750] = filterRGB(frames[600:750])
    #25-30
    frames[750:900] = filterHSV(frames[750:900])
    #30-35
    frames[900:1050] = filterHSV_morph(frames[900:1050])
    #39-43
    frames[1195:1320] = invisible_tower(frames[1195:1320], frames[1140])
    #42-45
    frames[1260:1350] = detect_faces(frames[1260:1350])
    #45-48
    frames[1350:1440] = change_color(frames[1350:1440])
    #48-52
    frames[1440:1560] = filterHSV(frames[1440:1560])
    #52-54
    frames[1560:1620] = detect_faces(frames[1560:1620])
    #54-60
    frames[1620:] = filterHSV(frames[1620:])



    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    out = cv2.VideoWriter('output.mp4', -1, 30, (480, 360))
    for frame in frames:
        frame = imutils.resize(frame, width=480, height=360)
        #cv2.imshow('Video', frame)
        out.write(frame)
        #if cv2.waitKey(30) & 0xFF == ord('q'):
         #   break

    out.release()
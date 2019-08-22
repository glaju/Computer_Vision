import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
from sklearn.metrics import mean_squared_error



def sobel_edge_detector(frames):
    for i in range(len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        sobel_vertical = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=2 * int(i / 45) + 1)
        sobel_horizontal = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=2 * int(i / 45) + 1)

        frames[i] = np.zeros((480, 640, 3), np.uint8)
        frames[i][:, :, 1] = sobel_vertical
        frames[i][:, :, 2] = sobel_horizontal

    return frames


def canny_edge_detector(frames):
    for i in range(len(frames)):
        frames[i] = cv2.Canny(frames[i], 50 + int(i / 2), 250 - int(i / 3), apertureSize=2 * int(i / 100) + 3)
        frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)

    return frames


def detect_circles(frames):
    for j in range(len(frames)):
        cimg = frames[j]
        img = cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=120+int(j/2), param2=40-int(j/10), minRadius=0, maxRadius=50)

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(frames[j], (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(frames[j], (i[0], i[1]), 2, (0, 0, 255), 3)
    return frames


def draw_rectangle(frames, p1, p2):
    for i in range(len(frames)):
        cv2.rectangle(frames[i], p1, p2, (0, 255, 0))
    return frames


def detect_object(frames):
    x = np.shape(frames[0])[1]
    y = np.shape(frames[0])[0]

    template_des = get_SIFT_features('./mug_image.jpg')
    step = 20
    x_window = 120
    y_window = 150

    for i in range(len(frames)):
        print i
        squared_error = np.zeros(((y - y_window) / step + 1, (x - x_window) / step + 1))
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        for j in range(y_window, y, step):
            for k in range(x_window, x, step):
                crop = gray[j - y_window:j + y_window, k - x_window:k + x_window]
                sift = cv2.SIFT(nfeatures=30, contrastThreshold=0.018, edgeThreshold=20, sigma=2)
                kp, des = sift.detectAndCompute(crop, None)

                try:

                    mse = ((template_des - des) ** 2).mean(axis=None)
                    squared_error[(j - y_window) / step, (k - x_window) / step] = mse
                except:

                    squared_error[(j - y_window) / step, (k - x_window) / step] = 0

        maximal = np.amax(squared_error)
        squared_error[np.where(squared_error == 0)] = maximal
        minimal = np.amin(squared_error)
        result = np.zeros((y, x), np.uint8)

        for j in range((y - step - y_window) / step):
            for k in range((x - step - x_window) / step):
                normalized = 255 - ((squared_error[j, k] - minimal) / (maximal - minimal) * 255)

                result[j * step:(j + 1) * step, k * step:(k + 1) * step] = int(normalized)

        frames[i] = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return frames


def get_SIFT_features(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT(nfeatures=30, contrastThreshold=0.018, edgeThreshold=20, sigma=2)
    kp, des = sift.detectAndCompute(gray, None)

    # img = cv2.drawKeypoints(gray, kp,  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return des


def template_matching(frames):
    for i in range(len(frames)):
        image = frames[i]
        template = cv2.imread('./mug_template.jpg')

        template = cv2.resize(template, (0, 0), fx=0.7, fy=0.7)

        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = min_loc
        h, w = templateGray.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 4)
        frames[i] = image

    return frames


def detect_object_SIFT(frames):
    template = cv2.imread('./mug_template.jpg')
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    for i in range(len(frames)):
        try:

            img2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()

            kp1, des1 = sift.detectAndCompute(template_gray, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            img = cv2.drawKeypoints(img2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            #cv2.imshow('img', img)
            #cv2.waitKey(0)

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1, des2, k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.95 * n.distance:
                    good.append(m)

            MIN_MATCH_COUNT = 8
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                h, w = template_gray.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                img2 = cv2.polylines(frames[i], [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
                frames[i] = img2
                frames[i] = np.zeros((480, 640), np.uint8)

                for j in range(len(dst_pts)):
                    y = int(dst_pts[j][0][0])
                    x = int(dst_pts[j][0][1])

                    frames[i][x - 10:x + 10, y - 10:y + 10] = 255

                frames[i] = cv2.GaussianBlur(frames[i], (15, 15), 0)
                frames[i] = cv2.GaussianBlur(frames[i], (15, 15), 0)
                frames[i] = cv2.GaussianBlur(frames[i], (15, 15), 0)
                # cv2.imshow('gray', frames[i])
                # cv2.waitKey(0)
            else:
                # print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
                matchesMask = None
                frames[i] = np.zeros((480, 640), np.uint8)
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)

            # img3 = cv2.drawMatches(template_gray, kp1, img2, kp2, good, None, **draw_params)

            # plt.imshow(img2, 'gray'), plt.show()

        # img = cv2.drawKeypoints(gray, kp,  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('img', img2)
        # cv2.waitKey(0)

        except:
            frames[i] = np.zeros((480, 640), np.uint8)
        frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)
    return frames


def detect_object_ORB(frames):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    template = cv2.imread('./mug_template.jpg')

    for i in range(len(frames)):
        im1Gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES, edgeThreshold=15, patchSize=15)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv2.drawMatches(template, keypoints1, frames[i], keypoints2, matches, None)
        cv2.imshow("matches.jpg", imMatches)
        cv2.waitKey(0)


if __name__ == '__main__':


    np.set_printoptions(threshold=sys.maxsize)
    cap = cv2.VideoCapture('./video2.mp4')

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



    # 5-10
    frames[150:300] = sobel_edge_detector(frames[150:300])
    # 10-15
    frames[300:450] = canny_edge_detector(frames[300:450])
    # 15-25
    frames[450:810] = detect_circles(frames[450:810])
    # 27-31
    frames[810:870] = draw_rectangle(frames[810:870], (270, 210), (380, 340))
    # 32-36
    frames[990:1020] = draw_rectangle(frames[990:1020], (30, 270), (230, 475))

    frames[870:930] = detect_object_SIFT(frames[870:930])
    frames[1020:1080] = detect_object_SIFT(frames[1020:1080])
    frames[1260:1350] = detect_object_SIFT(frames[1260:1350])

    frames[1400:1600] = template_matching(frames[1400:1600])
    frames[1700:1800] = detect_object_SIFT(frames[1700:1800])


    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output_2.mp4', fourcc, 30, (480, 360))
    # 0x00000021   *'a\0\0\0'  DIVX works with .avi
    # out = cv2.VideoWriter('./output4.avi', fourcc, 30, (640, 480))
    for frame in frames:
        frame = imutils.resize(frame, width=480, height=360)
        # cv2.imshow('Video', frame)
        out.write(frame)
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        #   break
    cv2.destroyAllWindows()
    out.release()

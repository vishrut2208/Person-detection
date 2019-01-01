import cv2
import numpy as np
from collections import namedtuple


FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
MIN_MATCH_COUNT = 10
flann_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)

matcher = cv2.FlannBasedMatcher(flann_params, {})
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
full_body = cv2.CascadeClassifier('haarcascade_fullbody.xml')

PlanarTarget = namedtuple('PlaneTarget', 'image, keypoints, descrs')
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad')


def detect_features(image):
    """
    extract ORB feature from Image
    :param image: Image
    :return: keypoints and descriptors list
    """
    detector = cv2.ORB_create(nfeatures=1000)
    keypoints, descrs = detector.detectAndCompute(image, None)
    if descrs is None:
        descrs = []
    return keypoints, descrs


def add_target():
    targets = []
    image = cv2.imread("tem_9.png")
    raw_points, raw_descrs = detect_features(image)
    points, descs = [], []
    for kp, desc in zip(raw_points, raw_descrs):
        points.append(kp)
        descs.append(desc)
    descs = np.uint8(descs)

    matcher.add([descs])
    target = PlanarTarget(image=image, keypoints=points, descrs=descs)
    targets.append(target)
    return targets


def track(img, targets):
    tracked = []
    frame_points, frame_descrs = detect_features(img)
    if len(frame_points) < MIN_MATCH_COUNT:
        return []
    matches = matcher.knnMatch(frame_descrs, k=2)
    matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
    if len(matches) < MIN_MATCH_COUNT:
        return []
    matches_by_id = [[] for _ in range(len(targets))]
    for m in matches:
        matches_by_id[m.imgIdx].append(m)
    for imgIdx, matches in enumerate(matches_by_id):
        if len(matches) < MIN_MATCH_COUNT:
            continue
        target = targets[imgIdx]
        p0 = [target.keypoints[m.trainIdx].pt for m in matches]
        p1 = [frame_points[m.queryIdx].pt for m in matches]
        p0, p1 = np.float32((p0, p1))
        H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
        status = status.ravel() != 0
        if status.sum() < MIN_MATCH_COUNT:
            continue
        p0, p1 = p0[status], p1[status]

        x0 = 13
        y0 = 13
        x1 = 300
        y1 = 300

        quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

        tracks = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
        tracked.append(tracks)
    tracked.sort(key=lambda t: len(t.p0), reverse=True)
    return tracked


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    targets = add_target()
    low_range = np.array([160, 90, 90])
    upper_range = np.array([180, 255, 255])
    object_height = 43
    while True:
        ret, frame = camera.read()
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.inRange(hsv_image, low_range, upper_range)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), thickness=3, lineType=cv2.LINE_4)
            pixel_permetric = h / object_height

        tracked = track(img, targets)

        for tr in tracked:
            x_min = int(min(tr.p1, key=lambda x: x[0])[0])
            x_max = int(max(tr.p1, key=lambda x: x[0])[0])
            y_min = int(min(tr.p1, key=lambda x: x[1])[1])
            y_max = int(max(tr.p1, key=lambda x: x[1])[1])

            # detect logo and draw white rectangle around it
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 256, 256), thickness=2)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            found, w = hog.detectMultiScale(img, winStride=(8, 8), padding=(32, 32), scale=1.05)
            for x, y, w, h in found:
                pad_w, pad_h = int(0.15*w), int(0.10*h)
                if x_min > (x + pad_w) and y_min > y + pad_h and x_max < x+w - pad_w and y_max < y+h - pad_h:

                    # detect logo and draw red rectangle around it
                    height = round(h / pixel_permetric)
                    height = "Height: " + str(height) + "cm"
                    print(height)
                    cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w, y + h), (255, 255, 0), thickness=2)
                    cv2.putText(img, height, (x, y+10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3, cv2.LINE_4)

            for (x, y, w, h) in faces:
                if x_min - int(0.25 * x) < x and x_max + int(0.25 * x) > x + w and (y_min < y + 20 or y_min > y - 10):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        # img = cv2.resize(img, (1920, 1024))
        cv2.imshow('plane', img)
        ch = cv2.waitKey(1)
        if ch == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

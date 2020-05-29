import cv2
import numpy as np
import sys

# provided by Alberto Ruiz from my university - http://dis.um.es/~alberto/
# pip install --upgrade http://robot.inf.um.es/material/umucv.tar.gz
from umucv.stream import autoStream
from umucv.util import ROI

bg = None
kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel used for closure operation


# Compute the running average for the background to later use as a chroma
def run_avg(image, weight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, weight)


# Get the contours of the hand (if any) and the segmented image
def segment(image, thresh=30):
    global bg
    # compute absolute difference between stored background and the current image
    diff = cv2.absdiff(bg.astype("uint8"), image)
    # threshold said difference
    thresh = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1]
    # apply a closing operation to fill in any gaps
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    # find the contours of the segmented hand
    (_, contours, _) = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # no contours no hand
    if len(contours) == 0:
        return
    else:
        segments = max(contours, key=cv2.contourArea)
        return closing, segments


# Main function
if __name__ == "__main__":
    cv2.namedWindow('input')
    roi = ROI('input')

    directory = sys.argv[1]     # Folder from where the dataset should be saved
    save = False                # Should we be saving files?
    calibrate = False           # Should we be calibrating?
    calibrated = False          # Is the background calibrated?
    calibration_time = 30       # Number of frames the program should calibrate the background
    accumulated_weight = 0.5    # self-explanatory
    num_frames = 0              # Current amount of frames used for the calibration process
    gesture = 0                 # Current gesture being saved
    image_number = 0            # Current image for said gesture being saved
    gesture_size = (200, 200)   # The selected ROI will be resized to said size

    # autoStream() utility could be replaced by cv.VideoCapture() and your usual loop:
    # cap = cv.VideoCapture(0)
    # while True:
    #     key = cv.waitKey(1) & 0xFF
    #     if key == 27: break
    #     ok, frame = cap.read()

    for key, frame in autoStream():
        clone = frame.copy()

        # umucv.roi utility is more annoying to replace and thus is left as an exercise to the reader
        if roi.roi:
            # get the ROI area
            [x1, y1, x2, y2] = roi.roi
            section = frame[y1:y2 + 1, x1:x2 + 1]

            # resize the ROI to our size of interest
            section_resized = cv2.resize(section, gesture_size)

            # begin or stop the calibration process with the 'c' key
            if key == ord('c'):
                if calibrate:
                    print(">>> Already calibrating, stopping now. ")
                    calibrate = False
                    calibrated = False
                    bg = None
                else:
                    print(">>> Starting...")
                    if image_number > 0:
                        gesture += 1
                        image_number = 0

                    num_frames = 0
                    calibrated = False
                    calibrate = True
                    save = False
            # start/stop generating the dataset by pressing the 's' key
            if key == ord('s'):
                if save:
                    # as long as the calibration is not reset, every time you start saving the file name will change from
                    # hand-(gesture)-(number).jpg to hand-(gesture+1)-(number).jpg
                    gesture += 1
                    image_number = 0
                    print(">>> No longer saving...")
                else:
                    print(">>> Saving now...")
                save = not save

            # calibrate the background if it hasn't been calibrated and the user pressed the 'c' key
            if calibrate and not calibrated:
                gray = cv2.cvtColor(section_resized, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                if num_frames < calibration_time:
                    run_avg(gray, accumulated_weight)
                    if num_frames == 1:
                        print(">>> Please wait! calibrating...")
                    elif num_frames == calibration_time - 1:
                        print(">>> Calibration successful...")
                        calibrated = True
                num_frames += 1
            elif calibrated:
                # transform the current ROI to grayscale and apply gaussian blur
                gray = cv2.cvtColor(section_resized, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                # do magic
                hand = segment(gray)
                # don't bother if there is nothing within the ROI
                if hand is not None:
                    threshold, segmented = hand
                    # show the threshold image
                    cv2.imshow("threshold", threshold)

                    if save:
                        file = directory + "hand" + str(gesture) + "-" + str(image_number) + ".jpg"
                        cv2.imwrite(file, threshold)
                        image_number += 1
                        print("saved: " + file)

            # Draw the ROI
            cv2.rectangle(clone, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
        cv2.imshow("input", clone)

# free up memory
cv2.destroyAllWindows()

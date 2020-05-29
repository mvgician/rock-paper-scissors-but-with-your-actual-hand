import os
import numpy as np
import cv2
import math
import time
from keras.models import model_from_json
import sys

# provided by Alberto Ruiz from my university - http://dis.um.es/~alberto/
# pip install --upgrade http://robot.inf.um.es/material/umucv.tar.gz
from umucv.stream import autoStream
from umucv.util import ROI, putText

from rps_bot import RockPaperScissorsBot
from rps_movement import MovementTracker

# comment this to enable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
bg = None
global loaded_model

# should match image_size from 'rps_train.py'
image_size = 50
# 3x3 kernel used for gaussian blur
kernel = np.ones((3, 3), np.uint8)
human_wins = 0
bot_wins = 0


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


# Function - here is where the magic happens
def count(thresh):
    # resize the segmented image to fit the size used in the training of this model
    thresh = cv2.resize(thresh, (image_size, image_size))
    thresh = thresh.reshape(1, 1, image_size, image_size).astype('float32')
    thresh = thresh / 255
    prob = loaded_model.predict(thresh)

    # return the probabilities only if there is a match
    if prob.any() > .99995:
        return loaded_model.predict_classes(thresh)
    return


def magnitude(x, y):
    return math.sqrt(x*x + y*y)


# 'game' logic
def fight(human, bot):
    global human_wins
    global bot_wins
    if human == "R":
        if bot == "R":
            return "Human Rock vs Bot Rock - Tie! Swing again..."
        elif bot == "P":
            bot_wins += 1
            return "Human Rock vs Bot Paper - Bot wins! Swing again..."
        else:
            human_wins += 1
            return "Human Rock vs Bot Scissors - Human wins! Swing again..."
    elif human == "P":
        if bot == "R":
            human_wins += 1
            return "Human Paper vs Bot Rock - Human wins! Swing again..."
        elif bot == "P":
            return "Human Paper vs Bot Paper - Tie! Swing again..."
        else:
            bot_wins += 1
            return "Human Paper vs Bot Scissors - Bot wins! Swing again..."
    elif human == "S":
        if bot == "R":
            bot_wins += 1
            return "Human Scissors vs Bot Rock - Bot wins! Swing again..."
        elif bot == "P":
            human_wins += 1
            return "Human Scissors vs Bot Paper - Human wins! Swing again..."
        else:
            return "Human Scissors vs Bot Scissors - Tie! Swing again..."
    else:
        return "Invalid input, try again..."


# Main function
if __name__ == "__main__":

    # load the model
    # expected format: python rps_cnn.py "rps_model.json" "rps_model_weights.h5"
    # we are all here grown-ups, no sanitizing required
    json_file = open(sys.argv[1], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(sys.argv[2])
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(">>> model successfully loaded")

    cv2.namedWindow('input')
    roi = ROI('input')

    calibrate = False  # Should we be calibrating?
    calibrated = False  # Is the background calibrated?
    calibration_time = 30  # Number of frames the program should calibrate the background
    accumulated_weight = 0.5  # self-explanatory
    num_frames = 0  # Current amount of frames used for the calibration process
    last_swing = 0  # The frame when the last swing was stored
    last_swing_mag = 0  # The magnitude of the last stored swing
    swing_counter = 0  # Number of swings accounted
    gesture_size = (200, 200)  # ROI region size
    human_choice = ""
    bot_choice = ""
    text = "Make a ROI and press C to calibrate."
    very_smart_bot = RockPaperScissorsBot()
    movement_tracker = MovementTracker()

    # autoStream() utility could be replaced by cv.VideoCapture() and your usual loop:
    # cap = cv.VideoCapture(0)
    # while True:
    #     key = cv.waitKey(1) & 0xFF
    #     if key == 27: break
    #     ok, frame = cap.read()

    for nf, (key, frame) in enumerate(autoStream()):
        t0 = time.time()

        # umucv.roi utility is more annoying to replace and thus is left as an exercise to the reader
        if roi.roi:
            # get the ROI area
            [x1, y1, x2, y2] = roi.roi
            section = frame[y1:y2 + 1, x1:x2 + 1]
            width = x2 - x1
            height = y2 - y1

            # resize the ROI to our size of interest
            section_resized = cv2.resize(section, gesture_size)

            # begin or stop the calibration process with the 'c' key
            if key == ord('c'):
                if calibrate:
                    print(">>> Already calibrating, stopping now. ")
                    text = "Calibration reset. Press C to calibrate again."
                    calibrate = False
                    calibrated = False
                    bg = None
                else:
                    print(">>> Starting...")
                    text = "Calibrating..."
                    num_frames = 0
                    calibrated = False
                    calibrate = True
                    save = False

            # calibrate the background if it hasn't been calibrated and the user pressed the 'c' key
            if calibrate and not calibrated:
                gray = cv2.cvtColor(section_resized, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)

                if num_frames < calibration_time:
                    run_avg(gray, accumulated_weight)
                    if num_frames == 1:
                        print(">>> Please wait! calibrating...")
                    elif num_frames == calibration_time - 1:
                        text = "Done! Swing your arm within the ROI"
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

                    average_velocity_x, average_velocity_y = movement_tracker.track_movement(threshold)
                    mag = magnitude(average_velocity_x, average_velocity_y)

                    # the swing detection is absolutely wonky and any improvement is more than welcome
                    # pretty much, whenever the magnitude of the motion detected in the current frame is high enough (30)
                    # and it has been updated (because the motion between two consecutive frames could be registered equally)
                    if mag > 30 and mag != last_swing_mag:
                        # then, a swing was detected
                        if last_swing == 0:
                            # First swing
                            swing_counter += 1
                        elif swing_counter == 2:
                            # Do nothing if you've accumulated enough attempts
                            text = "Got it, stop swinging!"
                        # consecutive swings should happen but, again, this is completely hardcoded
                        elif 50 > (nf - last_swing) > 3:
                            # There was a swing between 3 and 40 frames ago:
                            swing_counter += 1
                        last_swing_mag = mag
                        last_swing = nf

                    cv2.imshow("threshold", threshold)

                    # Whenever the user stops moving their hand after enough swings have been registered
                    if mag < 0.15 and swing_counter == 2:
                        # run the CNN model
                        category = count(threshold)
                        if category == 0:
                            human_choice = "R"
                        elif category == 1:
                            human_choice = "P"
                        elif category == 2:
                            human_choice = "S"
                        else:
                            human_choice = "N"
                        swing_counter = 0
                        bot_choice = very_smart_bot.output
                        # update the text on screen
                        text = fight(human_choice, bot_choice)

                        # only update the bot if a category was detected
                        if human_choice != "N":
                            very_smart_bot.update(human_choice)

            # Draw the ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)

        t1 = time.time()
        putText(frame, f'{(t1 - t0) * 1000:.0f} ms')
        putText(frame, text, orig=(5, 36))
        putText(frame, f'Human: {human_wins}, Machine: {bot_wins}', orig=(5, 56))
        cv2.imshow("input", frame)

cv2.destroyAllWindows()

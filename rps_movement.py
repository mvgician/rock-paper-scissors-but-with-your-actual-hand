# 'simple' optical flow script
import cv2
import numpy as np

class MovementTracker:
    def __init__(self):
        # Parameters for cv.goodFeaturesToTrack
        self.corners_params = dict(maxCorners=100,
                              qualityLevel=0.1,
                              minDistance=10,
                              blockSize=7)

        # Parameters for cv.calcOpticalFlowPyrLK
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # note: the usage of tracks is debatable, the optical flow between the last and current frame could be sufficient
        self.tracks = []  # Array with all the current detected points and their position through frames
        self.track_len = 2  # Number of frames stored of every detected point
        self.tracks_size = 50  # Max number of stored points
        self.detect_interval = 5  # Number of frames between point detection
        self.prev_frame = None
        self.n = 0  # Number of frames accounted thus far

    # returns the average velocity vector of all tracked points for the current frame
    def track_movement(self, cur_frame):
        # Average velocity vector of all detected points
        average_velocity_x = 0
        average_velocity_y = 0
        # Get the optical flow of our detected points using back-tracking for match verification between frames
        if len(self.tracks) > 0:
            # First, get the latest position of our points
            p0 = np.float32([t[-1] for t in self.tracks])

            # Compute the optical flow from the previous frame to the current one (1)
            p1, _, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, cur_frame, p0, None, **self.lk_params)

            # Then, repeat the same operation backwards to filter out invalid points (2)
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(cur_frame, self.prev_frame, p1, None, **self.lk_params)

            # Our criteria is that if the points detected in (2) return to their positions detected in (1), they are valid
            d = abs(p0 - p0r).reshape(-1, 2).max(axis=1)
            good = d < 1

            # Update the positions of all the valid points and discard the rest
            new_tracks = []
            for t, (x, y), ok in zip(self.tracks, p1.reshape(-1, 2), good):
                if not ok:
                    continue
                t.append([x, y])
                # Remove the oldest position if the limit has been reached
                if len(t) > self.track_len:
                    del t[0]
                new_tracks.append(t)
            self.tracks = new_tracks

        if len(self.tracks) > 0:
            # Get the average velocity of all points
            # It should be noted that this could be done during the drawing loop to cut an excess for loop
            for t in self.tracks:
                # We define the velocity of each point as the trajectory from the last known position to the current one
                # We also consider only those points with full known trajectory
                if len(t) == self.track_len:
                    x0, y0 = t[0]  # oldest stored position
                    x1, y1 = t[-1]  # newest stored position
                    average_velocity_x += x1 - x0
                    average_velocity_y += y1 - y0
            average_velocity_x /= len(self.tracks)
            average_velocity_y /= len(self.tracks)

        # Reset tracking every "detect_interval" frames
        if self.n % self.detect_interval == 0:
            # Mask out the current points to avoid repetition
            mask = np.zeros_like(cur_frame)
            mask[:] = 255
            for x, y in [np.int32(t[-1]) for t in self.tracks]:
                # We do that by adding circles to the mask corresponding to the last position of our stored points
                cv2.circle(mask, (x, y), 5, 0, -1)

            corners = cv2.goodFeaturesToTrack(cur_frame, mask=mask, **self.corners_params)
            # If any new points were found, add them as long as our capacity hasn't been reached
            if corners is not None:
                for [(x, y)] in corners.astype(np.float32):
                    if len(self.tracks) >= self.tracks_size:
                        break
                    self.tracks.append([[x, y]])
        self.prev_frame = cur_frame
        self.n += 1
        return average_velocity_x, average_velocity_y

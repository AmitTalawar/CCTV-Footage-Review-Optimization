import numpy as np
import cv2

#get_background() function will return the background model of the video
def get_background(file_path):
    cap = cv2.VideoCapture(file_path)
    #randomly select 50 frames for median
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT)*np.random.uniform(size=50)

    #store frames in array
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)

    median_frame = np.median(frames, axis=0).astype(np.uint8)
    return median_frame

#In the background model of the video we take median of 50 random frames
#In the median image or 'background model' only static objects remain
# we shall use this and subtract the current frames at frame differencing step

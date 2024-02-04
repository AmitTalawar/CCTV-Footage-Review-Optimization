import cv2, time, pandas
from datetime import datetime
import numpy as np
import pandas as pd


# Function to retrieve the background
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



# first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])

vid_path = "E:\VII Sem\MP II\Resources\VIRAT_S_000200_04_000937_001443.mp4"
# video=cv2.VideoCapture("VIRAT_S_000002.mp4")

# cv2.imshow("Median Frame",get_background(vid_path))
# cv2.waitKey(0)

video=cv2.VideoCapture(vid_path)

# Get the Background and Apply grayscale, gaussian blur
first_frame = get_background(vid_path)
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame=cv2.GaussianBlur(first_frame,(21,21),0)

# Retrieve Video Details 
total_frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)
end_time = total_frame_count/fps

# Initialize Start and End Motion Lists
start_t = []
end_t = []

while True:
    
    # Get current timestamp
    curr_frame_count = video.get(cv2.CAP_PROP_POS_FRAMES)
    curr_time = curr_frame_count/fps

    # Read Video
    check, frame = video.read()
    status=0
    if not check:
        break
    
    # Convert current frame to greyscale, gaussian blur
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

    # Get motion in current video using threshold
    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_frame=cv2.threshold(delta_frame, 15, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    # Adaptive threshold
    th2 = cv2.adaptiveThreshold(delta_frame,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,19,2)
    th3 = cv2.adaptiveThreshold(delta_frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,19,2)
    
    # Otsu Method
    th4 = cv2.threshold(delta_frame,15,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # Get contours
    # (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cnts,_)=cv2.findContours(th2.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # (cnts,_)=cv2.findContours(th3.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # (cnts,_)=cv2.findContours(th4.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    # For every contour, if above threshold, detect motion and create bounding box
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status=1
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
    status_list.append(status)

    # Retrieve last 2 elements of Status
    status_list=status_list[-2:]

    # Start => 0, 1 (No motion -> Motion)
    if status_list[-1]==1 and status_list[-2]==0:
        start_t.append(curr_time)

    # End => 1, 0 (Motion -> No motion)
    if status_list[-1]==0 and status_list[-2]==1:
        end_t.append(curr_time)

    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Global Threshold Frame",thresh_frame)
    cv2.imshow("Threshold Frame Mean",th2)
    cv2.imshow("Threshold Frame Gaussian",th3)
    cv2.imshow("Otsu Frame",th4)
    cv2.imshow("Color Frame",frame)

    key=cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        if status==1:
            times.append(datetime.now())
        break

print(start_t)
print(end_t)

# Handling Start and End time exceptions
# video starts with motion ends with motion
if len(start_t) == 0 and len(end_t) == 0:
    start_t.append(0.0)
    end_t.append(end_time)

# video starts with motion and stops before video ends and motion never restarts
if len(start_t) == 0 and len(end_t) == 1:
    start_t.append(0.0)

# Motion is somewhere after the video starts and the video ends with motion
if len(start_t) == 1 and len(end_t) == 0:
    end_t.append(end_time)

# Motion is from the begining and motion restarts throughout the video
if end_t[0] < start_t[0]:
    start_t.insert(0, 0.0)

# In case video ends with motion and there are multiple instances of motion detected
if len(start_t) != len(end_t):
    end_t.append(end_time)

# Rounding off the values to 2 decimal places
start_t = [round(i, 2) for i in start_t]
end_t = [round(i, 2) for i in end_t]

for i in range(len(start_t)):
    print(f"{start_t[i]} --> {end_t[i]}")


# For contiguous intervals for motion detection with gap lesser than 0.4 seconds we merge the intervals for better performance while object detection 
n_starts = []
n_ends = []
temp = 0
flag = 0
for i in range(len(start_t)):
    i = temp
    for j in range(1, len(start_t) - i):
        if start_t[i+j] - end_t[i] > 0.4:
            n_starts.append(start_t[i])
            n_ends.append(end_t[i+j-1])
            temp = i + j
            break
        else:
            if start_t[i+j] == start_t[-1] and flag == 0:
                n_starts.append(start_t[i])
                n_ends.append(end_t[-1]) 
                flag = 1

# Flag to avoid multiple similar iterations  
else:
    if flag == 0:
        n_starts.append(start_t[i])
        n_ends.append(end_t[-1])  

# Removing repititive last times
n_times = [(n_starts[i], n_ends[i]) for i in range(len(n_starts))]
n_times = list(set(n_times))
n_times.sort(key=lambda x: x[0])

for i in range(len(n_times)):
    n_starts[i] = n_times[i][0]
    n_ends[i] = n_times[i][1]

# n_starts = list(set(n_starts))
# n_ends = list(set(n_ends))

# For reference
# print(n_starts)
# print(n_ends)

# Converting lists to dataframe
df = pd.DataFrame({
    'Start Time' : n_starts, 
    'End Time' : n_ends
    })

# Write the times to CSV file
df.to_csv("Times.csv")
video.release()
cv2.destroyAllWindows


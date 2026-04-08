#this is the main file of the code where the genera loop will be running

import pyautogui
from collections import deque
import numpy as np
import time 

WINDOW_SIZE = 100   # e.g., 100 samples @ 50Hz = 2 seconds
STEP_SIZE   = 50    # 50% overlap → new prediction every 1 second


svm = #TBD
scaler = #TBD

def simulate_imu_stream(svm, scaler):
    buffer = deque(maxlen=WINDOW_SIZE)
    sample_count = 0

    while True:
        sample = read_imu_sensor()          # your hardware/socket call
        buffer.append(sample)
        sample_count += 1

        # predict every STEP_SIZE new samples once buffer is full
        if len(buffer) == WINDOW_SIZE and sample_count % STEP_SIZE == 0:
            window   = np.array(buffer)
            #features = extract_features(window) #WARNING if we extract features it may be too slow and mess things up.
            features = window 
            label    = svm.predict(scaler.transform([features]))[0] #should run in sub 1 ms
            print(f"[{time.time():.2f}] Activity: {label}")
            movement(label, buffer) 



#What we will probably do, is check every past 3 seconds of data. If the svm matches to something then we 
#will run the cooresponding command. 

captionsOn = False
activated = True

def movement(label, buffer):
    #Look Down - scroll down
    if(activated and label == "down"):
        pyautogui.scroll(-10)
    
    #Look Up - scroll up
    elif(activated and label == "up"):
        pyautogui.scroll(10)

    #Look left - less volume
    elif(activated and label == "left"):
        pyautogui.press('volumedown')


    #Look right - more volume
    elif(activated and label == "right"):
        pyautogui.press('volumeup')

    #Lay head on left shoulder - Play/Pause
    elif(activated and label == "left shoulder"):
        pyautogui.press('space')

    #Lay head on right shoulder - mute for youtube shorts
    elif(activated and label == "right shoulder"):
        pyautogui.press('m')

    #Activate close captions - Rotate head counterclockwise
    elif(activated and label == "CCW rotation"):
        if(captionsOn):
            pyautogui.click('C')
        else:
            pyautogui.click('c')
        captionsOn = not captionsOn

    #Rotate head clockwise - Activate/Deactivate key
    elif(activated and label == "CW rotation"):
        activated = not activated

    if(label != "None"):
        buffer.popleft()




if __name__ == "__main__":
    simulate_imu_stream(svm, scaler)

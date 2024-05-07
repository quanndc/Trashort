# from time import sleep
# import RPi.GPIO as GPIO

# DIR = 24   # Direction GPIO Pin
# STEP = 18  # Step GPIO Pin
# CW = 1     # Clockwise Rotation
# CCW = 0    # Counterclockwise Rotation
# SPR = 48   # Steps per Revolution (360 / 7.5)

# GPIO.setmode(GPIO.BCM)
# GPIO.setup(DIR, GPIO.OUT)
# GPIO.setup(STEP, GPIO.OUT)
# GPIO.output(DIR, CW)

# step_count = SPR
# delay = .0208

# for x in range(step_count):
#     GPIO.output(STEP, GPIO.HIGH)
#     sleep(delay)
#     GPIO.output(STEP, GPIO.LOW)
#     sleep(delay)

# sleep(.5)
# GPIO.output(DIR, CCW)
# for x in range(step_count):
#     GPIO.output(STEP, GPIO.HIGH)
#     sleep(delay)
#     GPIO.output(STEP, GPIO.LOW)
#     sleep(delay)

# GPIO.cleanup()


#open camera with opencv

# import the opencv library 
import cv2 
import numpy as np
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Converting the input frame to grayscale
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   

    # Fliping the image as said in question
    gray_flip = cv2.flip(gray,1)

    # Combining the two different image frames in one window
    combined_window = np.hstack([gray,gray_flip])

    # Displaying the single window
    cv2.imshow("Combined videos ",combined_window)
    key=cv2.waitKey(1)

    if key==ord('q'):
        break
print(a)

  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

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
import os
import cv2 

from lobe import ImageModel
import time
from rpi_python_drv8825.stepper import StepperMotor
from time import sleep
from interval_timer import IntervalTimer


# GPIO setup
enable_pin = 4
step_pin = 18
dir_pin = 24
mode_pins = (21, 22, 27)
# Stepper motor setup
step_type = '1/32'
fullstep_delay = .005

# create object
motor = StepperMotor(enable_pin, step_pin, dir_pin, mode_pins, step_type, fullstep_delay)

# define a video capture object 
vid = cv2.VideoCapture(0) 

# Load the model
model = ImageModel.load('/home/pi/Lobe/model')
  
# while(True): 
      
    # Capture the video frame 
    # by frame 
ret, frame = vid.read() 
  
    #disable mirror effect
frame = cv2.flip(frame, 1)
    # zoom in the image
    # frame = frame[100:500, 100:500]
    # Display the resulting frame 
    # cv2.imshow('frame', frame)
    
    
    # capture image
    # for interval in IntervalTimer(7):
        #resize image to 300x300
        # frame = cv2.resize(frame, (400, 400))
        #save image
cv2.imwrite('/home/trashort/Pictures/default_background/default_background.jpg', cv2.resize(frame, (400, 400)))
        # cv2.imwrite('/home/trashort/Pictures/image.jpg', cv2.resize(frame, (400, 400)))
        # result = model.predict_from_file('/home/trashort/Pictures/image.jpg')
        # print(result.prediction)

        # if result.prediction == 'Organic':
        #     motor.enable(True)
        #     motor.run(200*8, True)
        #     motor.run(200*8, False)  
        #     # enables stepper driver
        #     motor.enable(False)
        #     break
        # else:
        #     motor.enable(True)
        #     motor.run(200*8, False)
        #     motor.run(200*8, True)           # enables stepper driver
            
        #     motor.enable(False)
   # disables stepper driver


# After the loop release the cap object 
# vid.release() 
# Destroy all the windows 
# cv2.destroyAllWindows() 

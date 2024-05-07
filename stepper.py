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
from lobe import ImageModel
import time
from rpi_python_drv8825.stepper import StepperMotor
from time import sleep
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
# motor.enable(True)
# for i in range(100):

    # motor.enable(True)     # enables stepper driver
    # motor.run(200*32, True)     # run motor 6400 steps clowckwise
    # total_microsteps = 200*32 * 1/32
    # print(total_microsteps)
    # print(i+1.8)
    # actual_rotation_angle = float(total_microsteps) * 1.8
    # print("quay" + str(actual_rotation_angle))
    #sleep(0.5)
    # motor.run(200*32, False)  
    # total_microsteps = 200*32 * 1/32
    # print(total_microsteps)
    # print(i+1.8)
# define a video capture object 
vid = cv2.VideoCapture(0) 

# Load the model
model = ImageModel.load('/home/pi/Lobe/model')
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    #disable mirror effect
    frame = cv2.flip(frame, 1)
    # Display the resulting frame 
    cv2.imshow('frame', frame) 

    # capture image
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite('../Pictures/image.jpg', frame)
        result = model.predict_from_file('../Pictures/image.jpg')
        print(result.prediction)
        motor.enable(True)        # enables stepper driver
        motor.run(200*32, True)
        motor.run(200*32, False)  
        motor.enable(False)        # disables stepper driver
        
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(2) & 0xFF == ord('q'): 
        break
  


# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

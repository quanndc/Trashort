#import libs
import cv2 
from lobe import ImageModel
from rpi_python_drv8825.stepper import StepperMotor
from interval_timer import IntervalTimer


# GPIO setup
enable_pin = 4
step_pin = 18
dir_pin = 24
mode_pins = (21, 22, 27)
# Stepper motor setup
step_type = '1/32'
fullstep_delay = .005

# create motor object
motor = StepperMotor(enable_pin, step_pin, dir_pin, mode_pins, step_type, fullstep_delay)

# define a video capture object 
vid = cv2.VideoCapture(0) 

# Load the model
model = ImageModel.load('/home/pi/Lobe/model')
  
while(True):       
    # Capture the video frame by frame 
    ret, frame = vid.read() 
    #disable mirror effect
    frame = cv2.flip(frame, 1)
    # capture image
    for interval in IntervalTimer(10):
        #load the default background
        background = cv2.imread('/home/trashort/Pictures/default_background/default_background.jpg')
        #resize the background to 400x400
        background = cv2.resize(background, (400, 400))
        #check if the background is the same as the default background
        # resize frame to 400x400
        frame = cv2.resize(frame, (400, 400))
        if cv2.subtract(background, frame).mean() < 2:
            print('Background is the same as default background')
            continue
        else:
            cv2.imwrite('/home/trashort/Pictures/image.jpg', frame)
            # predict the image
            result = model.predict_from_file('/home/trashort/Pictures/image.jpg')
            print(result.prediction)

            #turn the stepper
            if result.prediction == 'Organic':
                motor.enable(True)
                motor.run(200*8, True)
                motor.run(200*8, False)  
                # enables stepper driver
                motor.enable(False)
                break
            else:
                motor.enable(True)
                motor.run(200*8, False)
                motor.run(200*8, True)
                 # enables stepper driver
                motor.enable(False)


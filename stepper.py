import RPi.GPIO as GPIO

# import the library
from RpiMotorLib import RpiMotorLib
    
#define GPIO pins
GPIO_pins = (21, 22, 27) # Microstep Resolution MS1-MS3 -> GPIO Pin
direction= 20       # Direction -> GPIO Pin
step = 21      # Step -> GPIO Pin

# Declare a instance of class pass GPIO pins numbers and the motor type
mymotortest = RpiMotorLib.A4988Nema(direction, step, GPIO_pins, "DRV8825")


# call the function, pass the arguments
mymotortest.motor_go(False, "Full" , 100, .01, False, .05)
from rpi_python_drv8825.stepper import StepperMotor
from time import sleep
# GPIO setup
enable_pin = 4
step_pin = 18
dir_pin = 24
mode_pins = (21, 22, 27)
# Stepper motor setup
step_type = '1/64'
fullstep_delay = .005


# create object
motor = StepperMotor(enable_pin, step_pin, dir_pin, mode_pins, step_type, fullstep_delay)
motor.enable(True)
for i in range(100):
    
    motor.enable(True)        # enables stepper driver
    motor.run(200*32, True)     # run motor 6400 steps clowckwise
    total_microsteps = 200*32 * 1/32
    print(total_microsteps)
    actual_rotation_angle = float(total_microsteps) * 1.8
    print("quay" + str(actual_rotation_angle))
    #sleep(0.5)
    motor.run(200*32, False)  
    total_microsteps = 200*32 * 1/32
    print(total_microsteps)
    actual_rotation_angle = float(total_microsteps) * 1.8
    print("quay" + str(actual_rotation_angle))  # run motor 6400 steps counterclockwise
    #motor.enable(False)       # disable stepper driver

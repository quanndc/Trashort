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

# import libs
from rpi_python_drv8825.stepper import StepperMotor  # type: ignore
from interval_timer import IntervalTimer
from tflite_runtime.interpreter import Interpreter  # type: ignore
import numpy as np
import cv2 as cv
import requests
import time
import os

url = "http://10.106.21.113:1323/"

# GPIO setup
enable_pin = 4
step_pin = 18
dir_pin = 24
mode_pins = (21, 22, 27)
# Stepper motor setup
step_type = "1/32"
fullstep_delay = 0.005
# create motor object
motor = StepperMotor(
    enable_pin, step_pin, dir_pin, mode_pins, step_type, fullstep_delay
)

# define a video capture object
# vid = cv2.VideoCapture(0)

# project folder
project_folder = "/home/trashort/Repos/Trashort/"
# model path
model_path = project_folder + "model.tflite"
label_path = project_folder + "labels.txt"


def load_labels(path):  # Read the labels from the text file as a Python list.
    with open(path, "r") as f:
        return [line.strip() for i, line in enumerate(f.readlines())]


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)

    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details["index"]))

    scale, zero_point = output_details["quantization"]
    output = scale * (output - zero_point)

    ordered = np.argpartition(-output, 1)
    return [(i, output[i]) for i in ordered[:top_k]][0]


def get_miliseconds():
    return int(time.time() * 1000)


model = Interpreter(model_path)
model.allocate_tensors()
_, height, width, _ = model.get_input_details()[0]["shape"]

for interval in IntervalTimer(10):
    vid = cv.VideoCapture(0)
    ret, frame = vid.read()
    # disable mirror effect
    frame = cv.flip(frame, 1)
    # capture image
    # resize the frame to 224x224
    frame = cv.resize(frame, (224, 224), interpolation=cv.INTER_AREA)
    # get current time
    current_time = get_miliseconds()
    # save the image
    cv.imwrite("/home/trashort/Pictures/" + str(current_time) + ".jpg", frame)
    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
    # Normalize the image array
    # image = (image / 127.5) - 1
    result, prob = classify_image(model, image)
    labels = load_labels(label_path)
    classification_label = labels[result]
    # print(classification_label)
    print(result)
    print(prob)
    # turn the stepper
    if result == 0:
        print("Background")
        os.remove("/home/trashort/Pictures/" + str(current_time) + ".jpg")
        vid.release()
        # continue
    elif result == 1:
        print("Recycle waste")
        motor.enable(True)
        motor.run(200 * 8, False)
        motor.run(200 * 8, True)
        # enables stepper driver
        motor.enable(False)
        with open("/home/trashort/Pictures/" + str(current_time) + ".jpg", "rb") as f:
            img_data = f.read()
            img_name = str(current_time) + ".jpg"
        r = requests.post(url + "uploadRecycle", files={"file": img_data})
        if r.status_code == 201:
            print("upload recycle waste")
            # turn off the camera
            os.remove("/home/trashort/Pictures/" + str(current_time) + ".jpg")
            vid.release()
    else:
        print("Organic waste")
        motor.enable(True)
        motor.run(200 * 8, True)
        motor.run(200 * 8, False)
        # enables stepper driver
        motor.enable(False)
        with open("/home/trashort/Pictures/" + str(current_time) + ".jpg", "rb") as f:
            img_data = f.read()
            img_name = str(current_time) + ".jpg"
        r = requests.post(url + "uploadOrganic", files={"file": img_data})
        if r.status_code == 201:
            print("upload organic waste")
            # turn off the camera
            os.remove("/home/trashort/Pictures/" + str(current_time) + ".jpg")
            vid.release()

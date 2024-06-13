# import libs
import cv2
from lobe import ImageModel
from rpi_python_drv8825.stepper import StepperMotor
from interval_timer import IntervalTimer
from tflite_runtime.interpreter import Interpreter
import numpy as np
import time
import cv2 as cv
import os

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


def checkBackground(img1, img2):
    # img1 = cv.imread("/home/trashort/Pictures/image.jpg",cv.IMREAD_GRAYSCALE)  # queryImage
    # img2 = cv.imread("/home/trashort/Pictures/default_background/image.jpg",cv.IMREAD_GRAYSCALE)  # trainImage

    

    # show img1
    # plt.imshow(img1)
    # plt.show()

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #print the points difference between the two images
    # print(matches[0].queryIdx, matches[0].trainIdx)
    # print(kp1[matches[1].queryIdx].pt, kp2[matches[1].trainIdx].pt)

    totalDistance = 0
    avaregeDistance = 0
    #calculate all distances between the two images
    for i in range(len(matches)):
        print(kp1[matches[i].queryIdx].pt, kp2[matches[i].trainIdx].pt)
        #calculate the distance between the two points
        distance = (np.linalg.norm(np.array(kp1[matches[i].queryIdx].pt) - np.array(kp2[matches[i].trainIdx].pt)))
        totalDistance += distance

    avaregeDistance = totalDistance / len(matches)
    print(avaregeDistance)
    return avaregeDistance


model = Interpreter(model_path)
model.allocate_tensors()
_, height, width, _ = model.get_input_details()[0]["shape"]

for interval in IntervalTimer(10):
    
    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()
    # disable mirror effect
    frame = cv2.flip(frame, 1)
    # frameForCheck = frame
    # capture image
    # cv2.imwrite("/home/trashort/Pictures/image.jpg", frame)
    # pic = cv2.imread("/home/trashort/Pictures/image.jpg")
    # get background image
    background = cv2.imread("/home/trashort/Pictures/default_background/image.jpg")
    
    padding_left = 100
    padding_right = 160
    background = background[:, padding_left : background.shape[1] - padding_right]
    frame = frame[:, padding_left : frame.shape[1] - padding_right]
    
    
    # resize the background to 400x400
    # background = cv2.resize(background, (224, 224), interpolation=cv2.INTER_AREA)
    # check if the background is the same as the default background
    # diffPoints = 0
    # print(background.shape)
    # print(frame.shape)
    diffPoints = checkBackground(background, frame)
    print(diffPoints)
    # os.remove("/home/trashort/Pictures/image.jpg")
    if diffPoints < 1:
        print("Background is the same as default background")
        # os.remove("/home/trashort/Pictures/image.jpg")
        vid.release()
        # continue
    else:
        # resize the frame to 224x224
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
        # Normalize the image array
        image = (image / 127.5) - 1
        result, prob = classify_image(model, image)
        labels = load_labels(label_path)
        classification_label = labels[result]
        # cv2.imwrite("/home/trashort/Pictures/image.jpg", frame)
        # predict the image
        # result = model.predict_from_file("/home/trashort/Pictures/image.jpg")
        # print(result.prediction)

        # turn the stepper
        if result == 0:
            print("Organic waste")
            motor.enable(True)
            motor.run(200 * 8, True)
            motor.run(200 * 8, False)
            # enables stepper driver
            motor.enable(False)
            vid.release()

        else:
            print("Recycle waste")
            motor.enable(True)
            motor.run(200 * 8, False)
            motor.run(200 * 8, True)
            # enables stepper driver
            motor.enable(False)
            vid.release()

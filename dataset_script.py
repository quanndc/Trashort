import requests
import cv2
import time
import os
from interval_timer import IntervalTimer

url = "http://172.20.255.198:1323/"


def get_miliseconds():
    return int(time.time() * 1000)


for interval in IntervalTimer(10):
    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()
    # disable mirror effect
    frame = cv2.flip(frame, 1)
    # show the frame
    #   cv2.imshow("frame", frame)

    # Capture the video frame by frame
    print("Capturing image...")

    # capture image
    # get the current time
    current_time = get_miliseconds()
    # cv2.imwrite(str(current_time) + ".jpg", frame)
    cv2.imwrite("/home/trashort/Pictures/" + str(current_time) + ".jpg", frame)
    with open("/home/trashort/Pictures/" + str(current_time) + ".jpg", "rb") as f:
        img_data = f.read()
        img_name = str(current_time) + ".jpg"

    r = requests.post(url + "uploadOrganic", files={"file": img_data})
    os.remove("/home/trashort/Pictures/" + str(current_time) + ".jpg")
    if r.status_code == 201:
        print("Image uploaded successfully!")
        # turn off the camera
        vid.release()

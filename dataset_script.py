import requests
import cv2
import time
import os

url = "http://localhost:1323/"
vid = cv2.VideoCapture(1)


def get_miliseconds():
    return int(time.time() * 1000)


while True:
  ret, frame = vid.read()
        # disable mirror effect
  frame = cv2.flip(frame, 1)
        # show the frame
#   cv2.imshow("frame", frame)
  if cv2.waitKey(1) & 0xFF == ord("c"):
        # Capture the video frame by frame
        

    # capture image
    # get the current time
    current_time = get_miliseconds()
    # cv2.imwrite(str(current_time) + ".jpg", frame)
    cv2.imwrite("/home/trashort/Pictures/" + str(current_time) + '.jpg', frame)
    with open("/home/trashort/Pictures/"+str(current_time)+ '.jpg', "rb") as f:
        img_data = f.read()
        img_name = str(current_time) + '.jpg'

    r = requests.post(url + "uploadRecycle", files={"file": img_data})
    os.remove("/home/trashort/Pictures/" + str(current_time) + '.jpg')
    if r.status_code == 201:
        print("Image uploaded successfully!")
        
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break
# After the loop release the cap object
# vid.release()
# Destroy all the windows
# cv2.destroyAllWindows()

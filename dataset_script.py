import pyrebase
import time


firebaseConfig = {
  apiKey: "AIzaSyBg8NoVuSXHBzoQ0i1W_V62oBI2B_PJHWU",
  authDomain: "trashort-26c32.firebaseapp.com",
  projectId: "trashort-26c32",
  storageBucket: "trashort-26c32.appspot.com",
  messagingSenderId: "261465509828",
  appId: "1:261465509828:web:6f964ac329833f06f2de91"
};

firebase = pyrebase.initialize_app(firebaseConfig)
bucket =  firebase.storage()




# post an image to bucket
def post_image(image_path):
    # post the image to the bucket
    bucket.child("images").put(image_path)
    # get the url of the image
    url = bucket.child("images").child(image_path).get_url(None)
    return url

post_image('/home/trashort/Pictures/default_background/default_background.jpg')
# Importing Libraries
import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image
from tensorflow.keras.utils import img_to_array
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from tensorflow import keras
import base64
from io import BytesIO

# load model
emotion_dict = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
classifier =load_model('model.h5')

# load weights into new model
classifier.load_weights("model.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def image_ip():
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])

    if image_file is not None:
        original_image = Image.open(image_file)
        original_image = np.array(original_image)

        gray=cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        faces= face_cascade.detectMultiScale(gray, 1.3, 3)
        for x,y,w,h in faces:
            sub_face_img=gray[y:y+h, x:x+w]
            resized=cv2.resize(sub_face_img,(48,48))
            normalize=resized/255.0
            reshaped=np.reshape(normalize, (1, 48, 48, 1))
            result=classifier.predict(reshaped)
            label=np.argmax(result, axis=1)[0]
            print(label)
            cv2.rectangle(original_image, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(original_image,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(original_image,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(original_image, emotion_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        return original_image


def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection", "Emotion Detection using Image"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Siddhi Naik, Prachi Channe, Kshitija Lade, Pratiksha Rale    
            Email : smnaik@mitaoe.ac.in , pmchanne@mitaoe.ac.in, kvlade@mitaoe.ac.in, pprale@mitaoe.ac.in""")
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV haarcascade model, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.

                 1. Real time face detection using web cam feed.

                 2. Real time face emotion recognization.

                 3. Face emotion recognization using input image.

                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    elif choice == "Emotion Detection using Image":
        img = image_ip()
        st.text("Emotion Detection")
        st.image(img)
    
    else:
        pass


if __name__ == "__main__":
    main()

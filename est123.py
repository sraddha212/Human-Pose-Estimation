import streamlit as st
import numpy as np
from PIL import Image
import cv2
import time
import random
import os

DEMO_IMAGE = 'stand.jpg'

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

width = 368
height = 368
inWidth = width
inHeight = height

# Check if the file exists in the specified path
model_path = "graph_opt.pb"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} does not exist. Please check the path.")

net = cv2.dnn.readNetFromTensorflow(model_path)

st.sidebar.title("Settings")
st.sidebar.text("Adjust parameters for pose estimation")

st.title("🎯 Human Pose Estimation with OpenCV 🎯")
st.text('Make sure you have a clear image with all parts visible!')

img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))

st.subheader('Original Image')
st.image(image, caption="Original Image", use_container_width=True)

thres = st.sidebar.slider('Threshold for detecting the key points', min_value=0, value=20, max_value=100, step=5)
thres = thres / 100

@st.cache_data
def poseDetector(frame, thres):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
    out = net.forward()
    out = out[:, :19, :, :]
    
    assert len(BODY_PARTS) == out.shape[1]
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)
    
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert partFrom in BODY_PARTS
        assert partTo in BODY_PARTS

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.line(frame, points[idFrom], points[idTo], color, 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, color, cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, color, cv2.FILLED)
            
    t, _ = net.getPerfProfile()
    
    return frame, t / cv2.getTickFrequency()

start_time = time.time()
output, execution_time = poseDetector(image, thres)
end_time = time.time()

st.subheader('Positions Estimated')
st.image(output, caption="Positions Estimated", use_container_width=True)

st.markdown(f'**Execution time:** {execution_time:.2f} seconds')

st.markdown('---')

st.sidebar.markdown('**Save Result**')
if st.sidebar.button('Save Image'):
    output_image = Image.fromarray(output)
    output_image.save('pose_estimation_result.jpg')
    st.sidebar.markdown('Image saved as `pose_estimation_result.jpg`')

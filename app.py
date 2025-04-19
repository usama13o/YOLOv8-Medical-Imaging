import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import detection.detect as detect
import classification.classify as classify
import segmentation.segment as segment




def train_models():
    detect.train()
    print("[INFO] Training Detection model done!")
    classify.train()
    print("[INFO] Training Classification model done!")
    
    # RUN THE FOLLOWING FOR PREPERING INPUT DATA FOR TRAINIG SEGMENTATION MODEL 
    segment.prepare_input()    
    segment.train()
    print("[INFO] Training Segmentation model done!")
    
    
    
def main():
    
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")
    st.markdown(
    """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:300px;
            margin-left:-300px;
        }
        </style>
    """,
    unsafe_allow_html=True,
    )
    
    app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Object Detection', 'Object Classification', "Object Segmentation"])
    
    
    if app_mode == 'About App':
        
        
        
        st.markdown("<style> p{margin: 10px auto; text-align: justify; font-size:20px;}</style>", unsafe_allow_html=True)      

        st.markdown("""
        # Welcome to our AI Web App Demo 

        This web demo application is designed to showcase the power of Artificial Intelligence (AI) in the realm of image processing. Our app offers three main features: **Image Classification**, **Detection**, and **Segmentation**. 

        ## Image Classification

        Image Classification is the task of assigning an input image one label from a fixed set of categories. This is one of the core tasks in Computer Vision that, despite its simplicity, has a large variety of practical applications.

        ## Detection

        Our app's detection feature identifies objects within images. It not only tells us what objects are present in the image, but also provides information about where in the image the objects are located with bounding boxes.

        ## Segmentation

        Image Segmentation is the process of partitioning an image into multiple segments. The goal is to change the representation of an image into something that is more meaningful and easier to analyze.

        Our AI-powered app is designed to be user-friendly and intuitive. Whether you're a student, a professional, or an AI enthusiast, this app will give you a hands-on experience of what AI can achieve in the field of image processing.

        We hope you enjoy using our app and find it informative and engaging. If you have any questions or feedback, please don't hesitate to reach out to us. Happy exploring!
        """)

    elif app_mode == "Object Detection":
        
        st.header("Object Detection: Cancer Detection",)
        
        st.sidebar.markdown("----")
        confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.35)
        
        img_file_buffer_detect = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=0)
        DEMO_IMAGE = "DEMO_IMAGES/BloodImage_00000_jpg.rf.5fb00ac1228969a39cee7cd6678ee704.jpg"
        
        if img_file_buffer_detect is not None:
            img = cv.imdecode(np.fromstring(img_file_buffer_detect.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer_detect))
        else:
            img = cv.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        # predict
        detect.predict(img, confidence, st)
        
    elif app_mode == "Object Classification":
        
        st.header("Classification of X-Ray Images")
        
        st.sidebar.markdown("----")
        
        img_file_buffer_classify = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=1)
        DEMO_IMAGE = "DEMO_IMAGES/094.png"
        
        if img_file_buffer_classify is not None:
            img = cv.imdecode(np.fromstring(img_file_buffer_classify.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer_classify))
        else:
            img = cv.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        # predict
        classify.predict(img, st)
        
    elif app_mode == "Object Segmentation":
        
        
        st.header("Segmentation of Medical Images")
        
        st.sidebar.markdown("----")
        
        img_file_buffer_segment = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=2)
        DEMO_IMAGE = "DEMO_IMAGES/benign (2).png"
        
        if img_file_buffer_segment is not None:
            img = cv.imdecode(np.fromstring(img_file_buffer_segment.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer_segment))
        else:
            img = cv.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        # predict
        segment.predict(img, st)
        
        
        
       
        


if __name__ == "__main__":
    try:
        
        # RUN THE FOLLOWING ONLY IF YOU WANT TO TRAIN MODEL AGAIN 
        # train_models()
        
        main()
    except SystemExit:
        pass
        


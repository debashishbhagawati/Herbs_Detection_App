import streamlit as st
import tempfile
from PIL import Image
import numpy as np
import cv2
from Detection_model import load_image_model, load_video_model, give_classnames

def main():
    st.title("Identification of Different Medicinal Plants")
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")

    app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Run on Image', 'Run on Video'])

    if app_mode == "About App":
        st.markdown(
            'Welcome to our  Herb&Seek application, the ultimate solution to your exploration needs of the botanical world! Our application provides you with an effortless identification of various medicinal plants. Take a snap or a video of the plant you are curious about and our application will recognise it within seconds. Customise your search with the varied options available.')
        st.image('media/herbal-medicinal-Plants.jpg')

    if app_mode == "Run on Image":
        st.sidebar.markdown('---')
        l = give_classnames()
        confidence = st.sidebar.slider('Confidence', value=0.08, min_value=0.0, max_value=1.0)
        select_mode_img = st.sidebar.multiselect('Choose', list(l.values()))
        st.sidebar.markdown('---')
        DEMO_IMAGE = 'media/Tamarind-scaled.jpg'

        img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

        if img_file_buffer is not None:
            img = cv2.imdecode(np.frombuffer(img_file_buffer.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer))
        else:
            img = cv2.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))

        st.sidebar.text('Original Image')
        st.sidebar.image(image)
        pred = load_image_model(img, confidence, select_mode_img)
        for prediction in pred:
            st.success(f"{prediction} is Detected")

    if app_mode == "Run on Video":
        st.sidebar.markdown('---')
        l = give_classnames()
        confidence_vid = st.sidebar.slider('Confidence', value=0.6, min_value=0.0, max_value=1.0)
        select_mode = st.sidebar.multiselect('Choose', list(l.values()))
        use_webcam = st.sidebar.checkbox('Use Webcam')
        st.sidebar.markdown('---')
        video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "nov", "asf"])

        DEMO_VIDEO = 'media/video.mp4'
        tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

        if not video_file_buffer:
            if use_webcam:
                tffile.name = 0
            else:

                tffile.name = DEMO_VIDEO
                demo_vid = open(tffile.name, 'rb')
                demo_bytes = demo_vid.read()
                st.sidebar.text('Input Video')
                st.sidebar.video(demo_bytes)
        else:
            tffile.write(video_file_buffer.read())
            demo_vid = open(tffile.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_bytes)

        stframe = st.empty()
        st.markdown("<hr/>", unsafe_allow_html=True)
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**Width**")
            kpi2_text = st.markdown("0")
        with kpi3:
            st.markdown("**Height**")
            kpi3_text = st.markdown("0")
        st.markdown("<hr/>", unsafe_allow_html=True)
        load_video_model(tffile.name, kpi1_text, kpi2_text, kpi3_text, stframe, confidence_vid, select_mode)


if __name__ == '__main__':
    main()

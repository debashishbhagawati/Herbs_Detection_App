import math
import cv2
from ultralytics import YOLO
import cvzone
import streamlit as st
import time

model_weights = 'best-nano.pt'


def load_image_model(img, confidence, select_mode):
    model = YOLO(model_weights)
    classnames = model.names
    result = list(model.predict(img, conf=confidence))[0]
    pred = []

    for r in result:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil(box.conf * 100) / 100
            cls = int(box.cls[0])
            if (classnames[cls] in select_mode) and (len(select_mode) != 0):
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=15, t=3, colorR=(135, 45, 73), rt=3)
                cvzone.putTextRect(img, pos=(x1, y1), text=f"{classnames[cls]} {conf}", colorR=(135, 45, 73), scale=1.5,
                                   thickness=2, offset=5)
                if classnames[cls] not in pred:
                    pred.append(classnames[cls])

            elif len(list(select_mode)) == 0:
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=15, t=3, colorR=(135, 45, 73), rt=3)
                cvzone.putTextRect(img, pos=(x1, y1), text=f"{classnames[cls]} {conf}", colorR=(135, 45, 73), scale=1.5,
                                   thickness=2, offset=5)
                if classnames[cls] not in pred:
                    pred.append(classnames[cls])

    st.subheader('Output Image')
    st.image(img, channels="BGR", use_column_width=True)
    return pred


def load_video_model(video_name, kpi1_text, kpi2_text, kpi3_text, st, confidence_vid, select_mode):
    cap = cv2.VideoCapture(video_name)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(model_weights)
    classnames = model.names

    prev_time = 0
    while True:
        res, frame = cap.read()

        if not res:
            break

        result = list(model.predict(frame, conf=confidence_vid, device="mps"))[0]
        pred = []
        for r in result:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                if (classnames[cls] in select_mode) and (len(select_mode) != 0):
                    cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=15, t=3, colorR=(135, 45, 73), rt=3)
                    cvzone.putTextRect(frame, pos=(x1, y1), text=f"{classnames[cls]} {conf}", colorR=(135, 45, 73),
                                       scale=1.5,
                                       thickness=2, offset=5)
                    if classnames[cls] not in pred:
                        pred.append(classnames[cls])

                elif len(list(select_mode)) == 0:
                    cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=15, t=3, colorR=(135, 45, 73), rt=3)
                    cvzone.putTextRect(frame, pos=(x1, y1), text=f"{classnames[cls]} {conf}", colorR=(135, 45, 73),
                                       scale=1.5,
                                       thickness=2, offset=5)
                    if classnames[cls] not in pred:
                        pred.append(classnames[cls])

        st.image(frame, channels="BGR", use_column_width=True)
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        kpi1_text.write(f"<h1 style='text-align:center; color:red;'>{fps:.1f}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{frame_width:.1f}</h1>", unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align:center; color:red;'>{frame_height:.1f}</h1>", unsafe_allow_html=True)
    cap.release()

    return pred


def give_classnames():
    model = YOLO(model_weights)
    classes = model.names
    return classes

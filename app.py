import os
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import time
from PoseModule import poseDetector, calculateAngle
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from pushup import push_up
from sample_utils.turn import get_ice_servers


st.title("Push Ups Counter")
st.subheader("An application that counts the number of push-ups you do in real-time.")
st.text("A framework called mediapipe is used to extract the pose landmarks\nAngle heuristics calculates the necessary joint angles and determines whether a\nproper push was performed.\nThe push-up counter increments by one if the conditions are met.")

show_fps = st.sidebar.radio(
    "Show fps:",
    ("Yes", "No")
)

model_complexity = int(st.sidebar.radio(
    "Model complexity:",
    ("0", "1", "2")
))

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


detector = poseDetector(model_complexity=model_complexity)
ptime = 0
push_up_count = 0
push_down = 0


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # img = push_up(frame)
    global ptime 
    global push_up_count 
    global push_down 
    global show_fps
    
    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (720, 480))
    # img = cv2.resize(img, (720, 720))
    img = detector.findPose(img)

    left_elbow = detector.findSpecificPosition(img, lm_ids = [11,13,15])
    right_elbow = detector.findSpecificPosition(img, lm_ids=[12,14,16]) 
    left_hip = detector.findSpecificPosition(img, lm_ids=[11,23,25])
    right_hip = detector.findSpecificPosition(img, lm_ids=[12,24,26])
    left_knee = detector.findSpecificPosition(img, lm_ids=[23,25,27])
    right_knee = detector.findSpecificPosition(img, lm_ids=[24,26,28])
    
    position_list = [left_elbow, right_elbow, left_hip, right_hip, left_knee, right_knee]
    
    if sum([len(i) for i in position_list]) == 18: 
        left_elbow_angle = int(calculateAngle(left_elbow[0], left_elbow[1], left_elbow[2]))
        right_elbow_angle = int(calculateAngle(right_elbow[0], right_elbow[1], right_elbow[2]))
        left_hip_angle = int(calculateAngle(left_hip[0], left_hip[1], left_hip[2]))
        right_hip_angle = int(calculateAngle(right_hip[0], right_hip[1], right_hip[2]))
        left_knee_angle = int(calculateAngle(left_knee[0], left_knee[1], left_knee[2]))
        right_knee_angle = int(calculateAngle(right_knee[0], right_knee[1], right_knee[2]))    
        
    if left_hip_angle in list(range(160, 210)) and right_hip_angle in list(range(160, 210)) and left_knee_angle in list(range(160, 210)) and right_knee_angle in list(range(160, 210)):
    
        if (left_elbow_angle >= 270 or left_elbow_angle <= 90) and (right_elbow_angle >= 270 or right_elbow_angle <= 90) and push_down == 0:
                push_down = 1
        if left_elbow_angle in list(range(160, 210)) and right_elbow_angle in list(range(160, 210)) and push_down == 1:
                push_up_count += 1    
                push_down = 0
                    
                    
        cv2.putText(img, "Push ups: " + str(push_up_count), (250,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)
    
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    #Uncomment this to display the fps.
    if show_fps == "Yes":
        cv2.putText(img, "fps: " + str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3) 

    # cv2.putText(img, "fps: ", (70,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)
    cv2.imshow("Image", img)
    print(push_up_count)
    # img = process(img)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="Pose estimation",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


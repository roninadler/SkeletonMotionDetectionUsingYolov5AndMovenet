from io import StringIO

import streamlit as st

st.title('Skeleton Motion Detection Using YOLOv5')
add_file_uploader_to_sidebar = st.sidebar.file_uploader("Upload video")

import tempfile
import DetectSkeletonOnVideo as Sk
import numpy as np
import cv2
import torch
import tensorflow as tf
import time
from pathlib import Path

@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True, show_spinner=False)
def load_model():
    st.spinner()
    with st.spinner(text='Loading YOLOv5 model...'):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        return model


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True, show_spinner=False)
def process_file(filename):
    st.spinner()
    with st.spinner(text='Processing video...'):
        frames = Sk.split_video_to_frames(tfile.name)
    return frames


def present_video(buf):
    stframe = st.empty()
    for i in range(len(buf)):
        curr_frame = cv2.cvtColor(buf[i], cv2.COLOR_BGR2RGB)
        stframe.image(curr_frame)
        time.sleep(0.05)
    stframe.empty()


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True, show_spinner=False)
def detect_video(frames):
    total_time_skeleton = 0
    total_time_yolo = 0
    total_time_draw_skeleton = 0

    out_video_path = 'out.avi'

    num_frames, image_height, image_width, _ = frames.shape
    crop_region = Sk.init_crop_region(image_height, image_width)

    buf = np.empty((frames.shape[0], frames.shape[1], frames.shape[2], 3), np.dtype('uint8'))
    index = 0

    model.classes = [0]

    tfile = tempfile.NamedTemporaryFile(delete=False)
    #tfile.write(add_file_uploader_to_sidebar.read())
    out1 = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (frames.shape[2], frames.shape[1]))
    out = cv2.VideoWriter(tfile.name, cv2.VideoWriter_fourcc(*"MJPG"), 25, (frames.shape[2], frames.shape[1]))

    my_bar = st.progress(0.0)
    stframe = st.empty()
    prog_counter = 0.0
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_original = frame.copy()
        xmin, ymin, xmax, ymax, yolo_time = Sk.yolo(frame, model)

        total_time_yolo += yolo_time
        frame_cropped = frame_original[ymin:ymax, xmin:xmax]

        image_height, image_width, _ = frame_cropped.shape
        crop_region = Sk.init_crop_region(image_height, image_width)

        image = tf.convert_to_tensor(frame_cropped, dtype=tf.float32)

        keypoints_with_scores, time_skeleton = Sk.run_inference(
            Sk.movenet, image, crop_region,
            crop_size=[input_size, input_size])

        total_time_skeleton += time_skeleton

        start = time.time()
        result = Sk.draw_prediction_on_image(
            image.numpy().astype(np.int32),
            keypoints_with_scores, crop_region=None,
            close_figure=True, output_image_height=frame_cropped.shape[0])
        end = time.time()
        total_time_draw_skeleton += (end - start)

        frame_original[ymin:ymax, xmin:xmax] = result
        if index % 10 == 0:
            stframe.image(frame_original)
        frame_original = cv2.cvtColor(frame_original, cv2.COLOR_RGB2BGR)
        buf[index] = frame_original
        index += 1
        prog_counter = float(index) / num_frames
        my_bar.progress(prog_counter)
        crop_region = Sk.determine_crop_region(
            keypoints_with_scores, image_height, image_width)
        out.write(frame_original)
        out1.write(frame_original)

    my_bar.empty()
    stframe.empty()
    out.release()
    out1.release()
    return buf, tfile



model = load_model()
input_size = 192
flag = False

if add_file_uploader_to_sidebar is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(add_file_uploader_to_sidebar.read())

    frames = process_file(tfile.name)

    buf, tfile_detected = detect_video(frames)

    show_video_button = st.sidebar.button('Show video')
    download_video_button = st.sidebar.download_button(label="Download video", data=tfile_detected.read(), file_name=f'out.avi')

    if show_video_button:
        present_video(buf)

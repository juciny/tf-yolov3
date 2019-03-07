#! /usr/bin/env python
# coding=utf-8

import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils


IMAGE_H, IMAGE_W = 416, 416
video_path = "./data/videos/0218.mp4"
classes = utils.read_coco_names('./data/video.names')
num_classes = len(classes)
input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(),
                                                            "./checkpoint/yolov3_cpu_nms.pb",
                                                          ["Placeholder:0", "concat_9:0", "mul_6:0"])
with tf.Session() as sess:
    vid = cv2.VideoCapture(video_path)
    FPS = round(vid.get(cv2.CAP_PROP_FPS))

    fps=vid.get(cv2.CAP_PROP_FPS)
    size=(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc=cv2.VideoWriter_fourcc('M','P','4','2')
    outVideo=cv2.VideoWriter("./result/0219_v1.0.avi",fourcc,fps,size)

    frame_num=0
    while True:
        return_value, frame = vid.read()
        if frame is None:
            break
        if frame_num/FPS>100:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
        img_resized = img_resized / 255.

        prev_time = time.time()

        boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)
        image = utils.draw_boxes(image, boxes, scores, labels, classes, (IMAGE_H, IMAGE_W), show=False)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255, 0, 0), thickness=2)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        outVideo.write(result)
        # cv2.imwrite("./result/0118/"+str(frame_num/FPS)+".jpg", result)
        if frame_num%FPS==0:
            print("det----",frame_num/FPS,"s")
        frame_num+=1
cv2.destroyAllWindows()

#! /usr/bin/env python
# coding=utf-8

import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils


IMAGE_H, IMAGE_W = 416, 416
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)

photo_path = "./data/demo_data/car.jpg"
image=cv2.imread(photo_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
img_resized = img_resized / 255.


gpu_nms_graph = tf.Graph()
# nms on GPU
input_tensor, output_tensors = utils.read_pb_return_tensors(gpu_nms_graph, "./checkpoint/yolov3_gpu_nms.pb",
                                           ["Placeholder:0", "concat_10:0", "concat_11:0", "concat_12:0"])

with tf.Session(graph=gpu_nms_graph) as sess:
    for i in range(5):
        start = time.time()
        boxes, scores, labels = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
        print("=> nms on gpu the number of boxes= %d  time=%.2f ms" %(len(boxes), 1000*(time.time()-start)))

    image = utils.draw_boxes(image, boxes, scores, labels, classes, (IMAGE_H, IMAGE_W), show=False)
    result = np.asarray(image)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./result/cat.jpg",result)

cv2.destroyAllWindows()

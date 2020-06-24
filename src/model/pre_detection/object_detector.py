import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
import time
import cv2
from src.data_manage.data_mange import DataMnage


class Object_Detector():
    def __init__(self, model_path):
        tf.compat.v1.reset_default_graph()
        self.data_manage = DataMnage()
        self._load_model(model_path)
        print('model loaded')

    def _load_model(self, model_path):
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with self.detection_graph.as_default():
            self.sess = tf.compat.v1.Session(config=config, graph=self.detection_graph)
            set_session(self.sess)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # load label_dict
        self.label_dict = {1: 'fish'}
        # warmup
        # self.detect_image(np.ones((600, 600, 3)))

    def do_train(self):
        # 세션을 열어 실제 학습을 진행합니다.
        with tf.Session() as sess:
            # 모든 변수들을 초기화한다.
            sess.run(tf.global_variables_initializer())


    def detect_image(self, image_path, score_thr=0.5, print_time=False, get_detect_info=None):
        image_np = self.data_manage.load_image(image_path)
        image_w, image_h = image_np.shape[1], image_np.shape[0]

        # Actual detection.
        t = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: np.expand_dims(image_np, axis=0)})
        if print_time:
            print('detection time :', time.time() - t)
        # Visualization of the resultsscores > score_thr of a detection.
        detection_info = {"file_name" : os.path.splitext(image_path)[0] + ".txt"}
        detection = ""
        for i, box in enumerate(boxes[scores > score_thr]):
            detection += self.data_manage.get_yolo_location(box, get_detect_info)
            top_left = (int(box[1] * image_w), int(box[0] * image_h))
            bottom_right = (int(box[3] * image_w), int(box[2] * image_h))
            cv2.rectangle(image_np, top_left, bottom_right, (0, 255, 0), 1)
            detection_name =  "{}-{}".format(self.label_dict[int(classes[0, i])],scores[0][i])
            cv2.putText(image_np,detection_name, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 1)
        detection_info["position"] = detection
        if get_detect_info is not None:
            return image_np, detection_info
        return image_np, _


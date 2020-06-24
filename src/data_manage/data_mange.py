import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
import time
import cv2
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class DataMnage():

    def __init__(self):
        pass

    def get_yolo_location(self, box, get_detect_info):
        top_left_x = str(round((box[3] + box[1])/2 , 6))
        top_left_y = str(round((box[2] + box[0])/2 , 6))
        bottom_right_x = str(round(box[3] - box[1], 6))
        bottom_right_y =str( round(box[2] - box[0], 6))
        result = "{} {} {} {} {}\n".format(get_detect_info, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        return result


    def load_image(self,image_path):
        img = cv2.imread(image_path)
        image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return image_np

    def write_yolo_tag(self,image_path, tag_info):
        write_path = os.path.splitext(image_path)[0] + ".txt"
        with open(write_path,"w",encoding="utf-8") as f:
            f.write(tag_info)
            f.close()

    def show_image(self,image_np):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(image_np)
        plt.show()
# import keras_retinanet
import argparse
import glob
import time

import numpy as np

from src.model.predict_model.data_process import DataProcess
from src.modules.retinanet.keras_retinanet import models
from src.modules.retinanet.keras_retinanet.utils.gpu import setup_gpu

class FishDetection():

    def __init__(self,model_path, labels_to_names, gpu=0,show_predict_img=False,score_thr=0.2):
        self.score_thr= score_thr
        self.show_predict_img = show_predict_img
        setup_gpu(gpu)
        self.model = models.load_model(model_path, backbone_name='resnet50')
        self.labels_to_names = labels_to_names
        self.data_process = DataProcess(labels_to_names)

    def predict(self,img_paths):
        for img_path in img_paths:
            image, scale, draw = self.data_process.img_processing(img_path)
            # process image
            start = time.time()
            boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes, scores, labels = self._get_pred_thr(boxes[0], scores[0], labels[0])
            print("processing time: ", time.time() - start)

            # correct for image scale
            print("img_path : {}".format(img_path))
            if len(boxes) != 0 :
                boxes /= scale
            boxes,scores, result = self.data_process.result_detection(draw,boxes,scores,labels,self.show_predict_img)
            # 나중에 반환값으로 사용하기

    def _get_pred_thr(self, pred_boxes, pred_scores, pred_labels):
        boxes, scores, labels = [], [], []
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            # scores are sorted so we can break
            if score < self.score_thr:
                break
            boxes.append(box)
            scores.append(score)
            labels.append(label)
        return np.array(boxes), scores, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test_dep')
    parser.add_argument('-data', help="학습 데이터 경로",default="./resources/images/test_img/")
    parser.add_argument('-extension', help="확장자",default="jpg")
    parser.add_argument('-model', help="학습 모델 경로", default="./snapshots/9class/resnet50_csv_25_infer.h5")
    parser.add_argument('-show_image', help="예측 여부", action='store_true',default=False)
    args = parser.parse_args()

    labels_to_names = {0: '감성돔', 1: '우럭', 2: '참돔', 3: '농어', 4 : "고등어", 5: "돌돔", 6 : "볼락",
                       7 : "숭어", 8 : "벤자리", 9 : "벵에돔",10:"광어",11:"노래미",12:"도다리",13:"삼치",
                       14:"쏨뱅이",15:"조기"}
    show_predict_img = True

    fish_detection = FishDetection(labels_to_names=labels_to_names,model_path=args.model
                                   ,show_predict_img=args.show_image)
    image_path = glob.glob(args.data + "*." + args.extension)
    fish_detection.predict(image_path)


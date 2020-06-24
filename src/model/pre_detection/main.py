import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from src.model.pre_detection.object_detector import Object_Detector
import argparse
import glob
from src.data_manage.data_mange import DataMnage

parser = argparse.ArgumentParser(description='test_dep')
parser.add_argument('-data', help="학습 데이터 경로")
parser.add_argument('-model', help="학습 모델 경로")
parser.add_argument('-show_image', help="예측 여부", action='store_true')
args = parser.parse_args()

# MODEL_PATH = './resources/model/fish_ssd_fpn_graph/frozen_inference_graph.pb'
MODEL_PATH = './resources/model/fish_inception_v2_graph/frozen_inference_graph.pb'
IMAGE_PATHS = "./resources/images/fish_image/bass_1/*.jpg"
get_detect_info = "3"
if args.data is not None:
    IMAGE_PATHS = args.data
if args.model is not None:
    MODEL_PATH = args.model

if __name__ == "__main__":
    run_model = Object_Detector(MODEL_PATH)
    data_manage = DataMnage()

    for image_path in glob.glob(IMAGE_PATHS):
        image_np, detection_info = run_model.detect_image(image_path,score_thr=0.85,print_time=True,get_detect_info=get_detect_info)
        data_manage.write_yolo_tag(image_path,detection_info["position"])
        if args.show_image:
            data_manage.show_image(image_np)
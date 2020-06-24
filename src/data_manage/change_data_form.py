import glob
from src.data_manage.data_mange import DataMnage
import os
import re

data_manage = DataMnage()
base_path = "./resources/images/obj/"
# base_path = "./resources/images/eval_obj/"

csv_file = open(base_path + "data.csv", "w", encoding="utf-8")
for class_name_path in glob.glob(base_path + "*"):
    class_name = os.path.basename(class_name_path)
    for file_path in glob.glob(class_name_path + "/*.txt"):
        img_path = os.path.splitext(file_path)[0]
        try:
            if os.path.isfile(img_path + ".jpg"):
                img_path = img_path + ".jpg"
                image_np = data_manage.load_image(img_path)
            elif os.path.isfile(img_path + ".png"):
                img_path = img_path + ".png"
                image_np = data_manage.load_image(img_path)
            elif os.path.isfile(img_path + ".gif"):
                img_path = img_path + ".gif"
                image_np = data_manage.load_image(img_path)
            else:
                raise Exception
        except:
            print("can't change file form : {}{} ".format(img_path, os.path.splitext(file_path)[1]))
            continue
        image_w, image_h = image_np.shape[1], image_np.shape[0]
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                img_path = class_name + "/" + os.path.split(img_path)[1]
                center_x = float(line.split()[1])
                center_y = float(line.split()[2])
                with_x = float(line.split()[3]) / 2
                with_y = float(line.split()[4]) / 2
                x1 = int((center_x - with_x) * image_w)
                y1 = int((center_y - with_y) * image_h)
                x2 = int((center_x + with_x) * image_w)
                y2 = int((center_y + with_y) * image_h)
                csv_file.write(
                    "{},{},{},{},{},{}\n".format(img_path, x1, y1, x2, y2, re.sub("_[0-9]{1,2}", "", class_name)))
            f.close()
csv_file.close()

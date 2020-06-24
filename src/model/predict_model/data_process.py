import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
print(plt.rcParams['font.family'])
import numpy as np
from PIL import Image

from src.modules.retinanet.keras_retinanet.utils.colors import label_color
from src.model.predict_model.object_data_mange import ObjectDataMnage

class DataProcess():

    def __init__(self,labels_to_names):
        self.object_data_manage = ObjectDataMnage()
        self.labels_to_names = labels_to_names

    def _draw_caption(self, image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    def result_detection(self, draw, boxes, scores, labels,show_plot=False):
        boxes, scores = self.object_data_manage.get_new_area(boxes, scores, labels)
        result_texts = []
        # visualize detections
        for box, score in zip(boxes, scores):
            max_label = score[0][0]
            color =  label_color(max_label)
            b = box.astype(int)
            self._draw_box(draw, b, color=color)
            result = ["{} : {:.2f}%".format(self.labels_to_names[s[0]], s[1]*100) for s in score]
            caption = ", ".join(result[:3])
            result_texts.append(caption)
            # caption = "{} {:.3f}".format(result)
            self._draw_caption(draw, b, caption)
            print("caption : {}".format(caption))
        if show_plot and len(boxes) != 0:
            plt.figure(figsize=(5, 5))
            plt.axis('off')
            plt.imshow(draw)
            plt.show()
        return boxes,scores,result_texts

    def img_processing(self,img_path):
        image = self._read_image_bgr(img_path)
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        image = self._preprocess_image(image)
        image, scale = self._resize_image(image)
        return image, scale, draw


    def _draw_box(self, image, box, color, thickness=2):
        """ Draws a box on an image with a given color.
        # Arguments
            image     : The image to draw on.
            box       : A list of 4 elements (x1, y1, x2, y2).
            color     : The color of the box.
            thickness : The thickness of the lines to draw a box with.
        """
        b = np.array(box).astype(int)
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

    def _read_image_bgr(self, path):
        image = np.asarray(Image.open(path).convert('RGB'))
        return image[:, :, ::-1].copy()

    def _preprocess_image(self, x, mode='caffe'):
        """ Preprocess an image by subtracting the ImageNet mean.
        Args
            x: np.array of shape (None, None, 3) or (3, None, None).
            mode: One of "caffe" or "tf".
                - caffe: will zero-center each color channel with
                    respect to the ImageNet dataset, without scaling.
                - tf: will scale pixels between -1 and 1, sample-wise.
        Returns
            The input with the ImageNet mean subtracted.
        """
        x = x.astype(np.float32)
        if mode == 'tf':
            x /= 127.5
            x -= 1.
        elif mode == 'caffe':
            x -= [103.939, 116.779, 123.68]
        return x

    def _resize_image(self, img, min_side=800, max_side=1333):
        """ Resize an image such that the size is constrained to min_side and max_side.

        Args
            min_side: The image's min side will be equal to min_side after resizing.
            max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

        Returns
            A resized image.
        """
        # compute scale to resize the image
        scale = self._compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)
        # resize the image with the computed scale
        img = cv2.resize(img, None, fx=scale, fy=scale)
        return img, scale

    def _compute_resize_scale(self,image_shape, min_side=800, max_side=1333):
        """ Compute an image scale such that the image size is constrained to min_side and max_side.
        Args
            min_side: The image's min side will be equal to min_side after resizing.
            max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.
        Returns
            A resizing scale.
        """
        (rows, cols, _) = image_shape
        smallest_side = min(rows, cols)
        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        return scale
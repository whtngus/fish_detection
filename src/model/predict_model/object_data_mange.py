import numpy as np

class ObjectDataMnage():

    def __init__(self,intersect_area_thrr=0.8):
        self.intersect_area_thrr = intersect_area_thrr

    def _compute_intersect_area(self,rect1, rect2):
        x1, y1 = rect1[0], rect1[1]
        x2, y2 = rect1[2], rect1[3]
        x3, y3 = rect2[0], rect2[1]
        x4, y4 = rect2[2], rect2[3]
        ## case1 오른쪽으로 벗어나 있는 경우
        if x2 < x3:
            return 0
        ## case2 왼쪽으로 벗어나 있는 경우
        if x1 > x4:
            return 0
        ## case3 위쪽 으로 벗어나 있는 경우
        if y2 < y3:
            return 0
        ## case4 아래쪽으로 벗어나 있는 경우
        if y1 > y4:
            return 0
        left_up_x = max(x1, x3)
        left_up_y = max(y1, y3)
        right_down_x = min(x2, x4)
        right_down_y = min(y2, y4)
        width = right_down_x - left_up_x
        height = right_down_y - left_up_y
        return width * height

    def get_new_area(self, boxes, scores, labels):
        if len(boxes) == 0:
            return boxes, scores
        new_boxes = []
        new_scores = []
        check_index = set()
        for index, (box, score,label) in enumerate(zip(boxes,scores,labels)):
            if index in check_index:
                continue
            score_tmp = {label:score/np.log(2)}
            check_index.add(index)
            new_boxes.append(box)
            for compair_index in range(index + 1, len(boxes)):
                if compair_index in check_index:
                    continue
                count = 3
                target_box = boxes[compair_index]
                target_label = labels[compair_index]
                target_score = scores[compair_index]
                interact_size = self._compute_intersect_area(box,target_box)
                box_min_size = min(self._compute_intersect_area(box,box),self._compute_intersect_area(target_box,target_box))
                if interact_size > box_min_size * self.intersect_area_thrr:
                    check_index.add(compair_index)
                    # score 보정
                    target_score /= np.log(count)
                    count += 1
                    if target_label in score_tmp:
                        score_tmp[target_label] += target_score
                    else:
                        score_tmp[target_label] = target_score
            new_scores.append(self._softmax(score_tmp))
        return new_boxes, new_scores

    def _softmax(self, score_map):
        datas = np.array([(k,v) for k,v in score_map.items()])
        exp_a = np.exp(datas[:,1])
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        result = {int(k):v for k,v in zip(datas[:,0],y)}
        return sorted(result.items(),key=(lambda x:x[1]),reverse=True)

#!/usr/bin/env python3 
import sys
import string
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import time
import boto3

class Cascade:
    def __init__(self, img_file, xml_filename, scale = 1.1, thresh = 0):
        self.stages = []
        self.imgname = img_file
        s3 = boto3.resource('s3')
        bucket = s3.Bucket('termproject18645')
        #load pre-trained classifier
        xml_file = bucket.Object(xml_filename)
        xml_file_body = xml_file.get()['Body'].read()
        root = ET.fromstring(xml_file_body)
        ele_features = root.findall('.//features')[0]

        for ele_stage in root.findall('.//stages')[0]:
            self.stages.append(Stage(ele_stage, ele_features))
        self.scale, self.thresh = scale, thresh

    def to_integral_image(self, img_arr):
        """
        Calculates the integral image based on this instance's original image data.
        img_arr: Original Image
        Return Integral image for given image
        """

        row_sum = np.zeros(img_arr.shape)
        integral_image_arr = np.zeros((img_arr.shape[0] + 1, img_arr.shape[1] + 1))
        for x in range(img_arr.shape[1]):
            for y in range(img_arr.shape[0]):
                row_sum[y, x] = row_sum[y - 1, x] + img_arr[y, x]
                integral_image_arr[y + 1, x + 1] = integral_image_arr[y + 1, x - 1 + 1] + row_sum[y, x]
        return integral_image_arr

    def sum_region(self, integral_img_arr, top_left, bottom_right):
        """
        Calculates the sum in the rectangle specified by the given tuples.
        integral_img_arr -- numpy.ndarray
        top_left: (x, y) of the rectangle's top left corner -- top_left: (int, int)
        bottom_right: (x, y) of the rectangle's bottom right corner -- bottom_right: (int, int)
        The sum of all pixels in the given rectangle -- int
        """

        top_left, bottom_right = tuple(top_left), tuple(bottom_right)

        top_right = (top_left[0], bottom_right[1])
        bottom_left = (bottom_right[0], top_left[1])

        return integral_img_arr[bottom_right] - integral_img_arr[top_right] - \
               integral_img_arr[bottom_left] + integral_img_arr[top_left]

    def _get_pos_scores(self, image):
        """
        From the given image,scales the image and 
        prints the positions of all subwindows matching the mapper's requirements
        and their corresponding scores
        Mapper only takes subwindows with a face and filters out overlapping subwindows
        """
        pos, scores = [], []
        w, h = image.size
        #compress if image too big
        ratio = 1.0 if w * h <= 5e4 else ((5e4 / (w * h)) ** 0.5)
        min_length = min(w, h)
        #repetitively scale the images to different sizes
        while min_length * ratio >= 24:
            img_resize = image.resize((int(w * ratio), int(h * ratio)))
            img_array = np.asarray(img_resize).astype(int)
            img_integ = self.to_integral_image(img_array).astype(int)
            img2_integ = self.to_integral_image(img_array ** 2).astype(int)
            cur_pos, cur_scores = self._get_cur_pos_scores(img_integ, img2_integ)
            for x, y in cur_pos:
                pos.append([x / ratio, y / ratio, 24 / ratio, 24/ratio])
            scores += list(cur_scores)
            ratio /= self.scale

        nms_pos, nms_score  = self._nms(pos,scores)
        #Hadoop only accepts tab seperator
        for position in nms_pos:
            print('%s\t%s' % (self.imgname, str(position).strip("[]").strip() ))

    def _get_cur_pos_scores(self, image, image2):
        """
        Divide an already scaled image into subwindows
        and returns all subwindows with a face andn their corresponding scores
        """
        height, width = image.shape
        num_pos = (height - 25) * (width - 25)

        pos = np.array([[x, y] for x in range(0, height - 25) \
								for y in range(0, width - 25)])

        N, n_pos = 24 * 24, pos.shape[0]

        img_arr, img_vars = np.zeros((25 * 25, n_pos), dtype = int), np.zeros(n_pos)

        for i in range(n_pos):

            x, y = pos[i]
            img_arr[:, i] = image[x:x+25, y:y+25].flatten()
            mean = self.sum_region(image, (x, y), (x+24, y+24)) / N

            img_vars[i] = max(0, self.sum_region(image2, (x, y), (x+24, y+24)) / N - (mean ** 2))
        img_vars = img_vars ** 0.5 + 1e-27

        n_stages = len(self.stages)


        for i, stage in enumerate(self.stages):

            pos, img_arr, img_vars, scores = stage.filter_pos(pos, img_arr, img_vars, self.thresh)


        return pos, scores
    def _nms(self, bboxes, scores):
	#%   Description: this function implements non maximum suppression (NMS),
	#%                a post-processing algorithm for merging the bounding boxes
	#%                that belong to the same face.
	#%                For each loop, it takes a bounding box with highest score,
	#%                and removes the rest of bounding boxes that overlap with it.
	#%
	#%   Input
	#%     bboxes: a collection of bounding boxes in the form of
	#%             [x1, y1, w1, h1;
	#%              x2, y2, w2, h2;
	#%                    ...
	#%              xm, ym, wm, hm]
	#%     scores: the scores corresponding to the bounding boxes
	#%
	#%   Output:
	#%     bboxes_nms: a collection of bounding boxes after NMS
	#%     scores_nms: a collection of scores after NMS
        bboxes_nms = []
        scores_nms = []
        while len(scores) > 0:
            ind = scores.index(max(scores))
            bboxes_nms.append(bboxes[ind])
            scores_nms.append(scores[ind])

            cur_bbox = bboxes.pop(ind)
            cur_score = scores.pop(ind)
            for index in self._overlap(cur_bbox, bboxes)[::-1]:
                bboxes.pop(index)
                scores.pop(index)
        return np.array(bboxes_nms), np.array(bboxes_nms)
    def _overlap(self, cur_bbox, bboxes):
    #%   Description: this function return the index of bounding boxes that overlap
    #%                with the one with highest score.
    #%
    #%   Input
    #%     cur_bbox: the bounding box with highest score
    #%     bboxes: a collection bounding boxes, which may or may not be removed
    #%
    #%   Output
    #%     index: the index of bounding boxes to be removed

        if not bboxes:
            return []
        else:
            x1 = cur_bbox[0]
            x2 = cur_bbox[0] + cur_bbox[2]
            y1 = cur_bbox[1]
            y2 = cur_bbox[1] + cur_bbox[3]

            bboxes = np.array(bboxes)
            X1 = bboxes[:, 0]
            X2 = bboxes[:, 0] + bboxes[:, 2]
            Y1 = bboxes[:, 1]
            Y2 = bboxes[:, 1] + bboxes[:, 3]

            # check overlap
            index = []
            for i in range(X1.size):
                isoverlap = ((x1<=X1[i]<=x2 or x1<=X2[i]<=x2) and \
                            (y1<=Y1[i]<=y2 or y1<=Y2[i]<=y2)) or \
                            ((X1[i]<=x1<=X2[i] or X1[i]<=x2<=X2[i]) and \
                            (Y1[i]<=y1<=Y2[i] or Y1[i]<=y2<=Y2[i]))
                if isoverlap:
                    index.append(i)
            return index

class Stage:
    #%   cascade stage
    def __init__(self, ele_stage, ele_features):

        self.maxWeakCount = int(ele_stage.findall('.//maxWeakCount')[0].text)

        self.stageThreshold = float(ele_stage.findall('.//stageThreshold')[0].text)

        weak_classifiers = ele_stage.findall('.//weakClassifiers')[0]

        self.thresholds = np.zeros(self.maxWeakCount)

        self.leafvalues = np.zeros((self.maxWeakCount, 2))

        self.featArray = np.zeros((self.maxWeakCount, 25 * 25))

        for i, ele_class in enumerate(weak_classifiers):

            internalNodes = ele_class.findall(".//internalNodes")[0].text.split()

            self.thresholds[i] = float(internalNodes[3])

            leafvalues = list(map(float, ele_class.findall(".//leafValues")[0].text.split()))

            self.leafvalues[i, :] = np.array(leafvalues)

            target_feature = ele_features[int(internalNodes[2])]

            self.featArray[i, :] = self._get_feat_array(target_feature)


    def filter_pos(self, pos, img_arr, img_vars, thresh):
        """
        Compute all feature scores for all sub-windows in an scaled image
        Remove the subwindows that do not contain a face
        """

        n_pos = pos.shape[0]

        feat_values = np.matmul(self.featArray, img_arr) / img_vars / (24 * 24)

        feat_values = (feat_values.T - self.thresholds).T

        feat_sum = np.zeros(feat_values.shape)

        for i in range(self.maxWeakCount):
            feat_sum[i, feat_values[i, :] >= 0] = self.leafvalues[i, 1]

            feat_sum[i, feat_values[i, :] < 0] = self.leafvalues[i, 0]

        scores = np.sum(feat_sum, axis = 0).flatten()

        indexs = np.argwhere(scores <= (-2 + thresh)).flatten()

        return np.delete(pos, indexs, 0), np.delete(img_arr, indexs, 1), \
				np.delete(img_vars, indexs, 0), np.delete(scores, indexs, 0)


    def _get_feat_array(self, ele_feat):
        """
        weak classifier features
        """
        rects = ele_feat.findall(".//rects")[0]

        feat = np.zeros(25 * 25)

        getPos = lambda pos : pos[0] * 25 + pos[1]

        for rect in rects:

            texts = list(map(int, rect.text[:-1].split()))

            pos_00 = tuple(texts[1::-1])

            pos_11 = (texts[1]+ texts[3], texts[0]+ texts[2])

            pos_01 = (pos_00[0], pos_11[1])

            pos_10 = (pos_11[0], pos_00[1])

            feat[getPos(pos_00)] += texts[4]

            feat[getPos(pos_11)] += texts[4]

            feat[getPos(pos_01)] -= texts[4]

            feat[getPos(pos_10)] -= texts[4]
        return feat

def mapper():
    xml_filename = 'haarcascade_frontalface_default.xml'
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('termproject18645')

    for line in sys.stdin:
        filename = line.strip()
        image_object = bucket.Object(filename)
        image_object.download_file(filename)
        image = Image.open(filename, 'r').convert('L')
        cascade = Cascade(filename, xml_filename, scale = 1.1, thresh = 0)
        faces = cascade._get_pos_scores(image)

if __name__ == "__main__":
	mapper()



import numpy as np
from PIL import Image
from HaarLikeFeature import *
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


class Cascade:

	def __init__(self, xml_file_path, scale = 1.1, thresh = 0):

		self.stages = []

		root = ET.parse(xml_file_path).getroot()

		ele_features = root.findall('.//features')[0]

		for ele_stage in root.findall('.//stages')[0]:

			self.stages.append(Stage(ele_stage, ele_features))

		self.scale, self.thresh = scale, thresh


	def set_paras(self, scale = 1.1, thresh = 0):

		self.scale, self.thresh = scale, thresh


	def get_faces(self, image):

		pos, scores = self._get_pos_scores(image)

		img_array, faces = np.asarray(image).astype(int), []

		for x, y, h, w in pos:

			faces.append(img_array[int(x):int(x+h), int(y):int(y+w)].copy())

		return faces


	def get_best_face(self, image):

		pos, scores = self._get_pos_scores(image)

		if len(pos) == 0:

			return []

		x, y, h, w = pos[list(scores).index(max(scores))]

		img_array = np.asarray(image).astype(int)

		face = img_array[int(x):int(x+h), int(y):int(y+w)].copy()

		return [face]


	def _get_pos_scores(self, image):

		pos, scores = [], []

		w, h = image.size

		ratio = 1.0 if w * h <= 5e4 else ((5e4 / (w * h)) ** 0.5)

		min_length = min(w, h)

		while min_length * ratio >= 24 :

			img_resize = image.resize((int(w * ratio), int(h * ratio)))

			img_array = np.asarray(img_resize).astype(int)
			
			img_integ = to_integral_image(img_array).astype(int)

			img2_integ = to_integral_image(img_array ** 2).astype(int)

			cur_pos, cur_scores = self._get_cur_pos_scores(img_integ, img2_integ)

			for x, y in cur_pos:

				pos.append([x / ratio, y / ratio, 24 / ratio, 24 / ratio])

			scores += list(cur_scores)

			print(f"Detecting face : { int(np.log(min_length * ratio / 24) / np.log(1.1)) } steps left.")

			ratio /= self.scale			
		
		return self._nms(pos, scores)


	def _get_cur_pos_scores(self, image, image2):

		height, width = image.shape

		num_pos = (height - 25) * (width - 25)

		pos = np.array([[x, y] for x in range(0, height - 25) \
								for y in range(0, width - 25)])

		N, n_pos = 24 * 24, pos.shape[0]

		img_arr, img_vars = np.zeros((25 * 25, n_pos), dtype = int), np.zeros(n_pos)

		for i in range(n_pos):

			x, y = pos[i]

			img_arr[:, i] = image[x:x+25, y:y+25].flatten()

			mean = sum_region(image, (x, y), (x+24, y+24)) / N

			img_vars[i] = max(0, sum_region(image2, (x, y), (x+24, y+24)) / N - (mean ** 2))

		img_vars = img_vars ** 0.5 + 1e-27

		n_stages = len(self.stages)

		#print("Start computing")

		for i, stage in enumerate(self.stages):

			pos, img_arr, img_vars, scores = stage.filter_pos(pos, img_arr, img_vars, self.thresh)

			#if n_stages > 2:

				#print(f"    { i * 1e4 // (n_stages - 1) / 1e2 } % : Finish stage{ i }.")
			
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
		return np.array(bboxes_nms), np.array(scores_nms)


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
	# image is the origin image

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

	

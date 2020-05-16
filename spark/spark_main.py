from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pyspark, sys, boto3

SCALE = 1.1
THRESH = 0.0
STAGES_BC = None
GROUP_SIZE_BC = 0


def to_integral_image(img_arr):
	"""
		Get the integral image.
		Integral Image:
		+ - - - - -        + -  -  -  -  -  -
		| 1 2 3 4 .        | 0  0  0  0  0  .
		| 5 6 7 8 .   =>   | 0  1  3  6 10  .
		| . . . . .        | 0  6 14 24 36  .
						   | .  .  .  .  .  .

	Calculates the integral image based on this instance's original image data.
	img_arr: Original Image
	Return Integral image for given image
	"""

	row_sum = np.zeros(img_arr.shape)
	integral_image_arr = np.zeros((img_arr.shape[0] + 1, img_arr.shape[1] + 1))
	for x in range(img_arr.shape[1]):
		for y in range(img_arr.shape[0]):
			row_sum[y, x] = row_sum[y-1, x] + img_arr[y, x]
			integral_image_arr[y+1, x+1] = integral_image_arr[y+1, x-1+1] + row_sum[y, x]
	return integral_image_arr

def sum_region(integral_img_arr, top_left, bottom_right):
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

def img_nms(bboxes, scores):
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
		for index in img_overlap(cur_bbox, bboxes)[::-1]:
			bboxes.pop(index)
			scores.pop(index)
	return np.array(bboxes_nms), np.array(scores_nms)

def img_overlap(cur_bbox, bboxes):
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

# get feature array
def get_featArray(ele_feat):
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

# get cascaded classifiers
def get_stages(xml_file):
	s3 = boto3.resource('s3')
	obj = s3.Object('image.collection', xml_file)
	
	stages = []
	root = ET.parse(obj.get()['Body']).getroot()
	ele_features = root.findall('.//features')[0]
	for ele_stage in root.findall('.//stages')[0]:
		cur_stage = dict()
		cur_stage['maxWeakCount'] = int(ele_stage.findall('.//maxWeakCount')[0].text)
		cur_stage['stageThreshold'] = float(ele_stage.findall('.//stageThreshold')[0].text)

		weak_classifiers = ele_stage.findall('.//weakClassifiers')[0]
		thresholds = np.zeros(cur_stage['maxWeakCount'])
		leafvalues = np.zeros((cur_stage['maxWeakCount'], 2))
		featArray = np.zeros((cur_stage['maxWeakCount'], 25 * 25))

		for i, ele_class in enumerate(weak_classifiers):
			internalNodes = ele_class.findall(".//internalNodes")[0].text.split()
			thresholds[i] = float(internalNodes[3])
			leafvalues[i, :] = np.array(list(map(float, ele_class.findall(".//leafValues")[0].text.split())))
			featArray[i, :] = get_featArray(ele_features[int(internalNodes[2])])
		
		cur_stage['thresholds'] = thresholds
		cur_stage['leafvalues'] = leafvalues
		cur_stage['featArray'] = featArray
		
		stages.append(cur_stage)

	return stages

# split image into subwindows
def img_2_windows(img_name):
	s3 = boto3.resource('s3')
	obj = s3.Object('image.collection', img_name)
	image = Image.open(obj.get()['Body'], 'r').convert('L')
	
	w, h = image.size
	# ratio = 1.0 if w * h <= 5e4 else ((5e4 / (w * h)) ** 0.5)
	ratio = 1.0
	min_length = min(w, h)
	res = []
	
	while min_length * ratio >= 24 :
		img_resize = image.resize((int(w * ratio), int(h * ratio)))
		img_array = np.asarray(img_resize).astype(int)
		img_integ = to_integral_image(img_array).astype(int)
		img2_integ = to_integral_image(img_array ** 2).astype(int)
		
		height, width = img_integ.shape
		num_pos = (height - 25) * (width - 25)
		pos = np.array([[x, y] for x in range(0, height - 25) \
								for y in range(0, width - 25)])

		N, n_pos = 24 * 24, pos.shape[0]
		img_arr, img_vars = np.zeros((25 * 25, n_pos), dtype = int), np.zeros(n_pos)
		
		for i in range(n_pos):
			x, y = pos[i]
			mean = sum_region(img_integ, (x, y), (x+24, y+24)) / N
			img_var = max(0, sum_region(img2_integ, (x, y), (x+24, y+24)) / N - (mean ** 2))
			img_vars[i] = img_var ** 0.5 + 1e-27
		res.append(((img_name, ratio), (pos, img_vars)))
		
		ratio /= SCALE	
		
	return res

# group subwindows
def group_windows(windows):
	img_name, ratio = windows[0]
	poses, img_vars = windows[1]
	size_group = GROUP_SIZE_BC.value
	
	if len(poses) < size_group:
		return [((img_name, ratio, 0), windows)]
	n_group = (len(poses) + size_group - 1) // size_group
	res = []
	for i in range(0, len(poses), size_group):
		group_id = (img_name, ratio, i)
		if i + size_group <= len(poses):
			group_val = ((img_name, ratio), \
						(poses[i:i+size_group], img_vars[i:i+size_group]))
		else:
			group_val = ((img_name, ratio), (poses[i:], img_vars[i:]))
		res.append((group_id, group_val))
	return res

# get list of windows from partition
def get_windows_from_iter(iter):
	windows = []
	s3 = boto3.resource('s3')
	imgs = dict()
	for block in list(iter):
		img_name, ratio = block[1][0]
		if img_name not in imgs:
			cur_img = dict()
			obj = s3.Object('image.collection', img_name)
			cur_img['image'] = Image.open(obj.get()['Body'], 'r').convert('L')
			imgs[img_name] = cur_img
		if ratio not in imgs[img_name]:
			image = imgs[img_name]['image']
			w, h = image.size
			img_resize = image.resize((int(w * ratio), int(h * ratio)))
			img_array = np.asarray(img_resize).astype(int)
			imgs[img_name][ratio] = to_integral_image(img_array).astype(int)
		poses, img_vars = block[1][1]
		for i in range(len(poses)):
			windows.append(((img_name, ratio, poses[i]), img_vars[i]))
	return windows, imgs

# apply cascaded classifiers to detect faces
def filter_windows(iter):
	windows, imgs = get_windows_from_iter(iter)
	n_pos = len(windows)
	img_arr = np.zeros((25 * 25, n_pos), dtype = int)	
	img_vars = np.zeros(n_pos)
	
	for i, entry in enumerate(windows):
		img_name, ratio, pos = entry[0]
		x, y = pos
		img_integ = imgs[img_name][ratio]
		img_arr[:, i] = img_integ[x:x+25, y:y+25].flatten()
		img_vars[i] = entry[1]
	
	scores = None
	remains = np.array([i for i in range(n_pos)])
	for i, stage in enumerate(STAGES_BC.value):
		feat_values = np.matmul(stage['featArray'], img_arr) / img_vars / (24 * 24)
		feat_values = (feat_values.T - stage['thresholds']).T
		feat_sum = np.zeros(feat_values.shape)
		for i in range(stage['maxWeakCount']):
			feat_sum[i, feat_values[i, :] >= 0] = stage['leafvalues'][i, 1]
			feat_sum[i, feat_values[i, :] < 0] = stage['leafvalues'][i, 0]
		scores = np.sum(feat_sum, axis = 0).flatten()
		indexs = np.argwhere(scores <= (-2 + THRESH)).flatten()
		
		remains = np.delete(remains, indexs, 0)
		img_arr = np.delete(img_arr, indexs, 1)
		img_vars = np.delete(img_vars, indexs, 0)
		scores = np.delete(scores, indexs, 0)
	
	res = []
	for i, index in enumerate(remains):
		img_name, ratio, pos = windows[index][0]
		box = [pos[0] / ratio, pos[1] / ratio, 24 / ratio, 24 / ratio]
		res.append((img_name, ([box], [scores[i]])))
	return res

if __name__ == "__main__":
	
	num_slaves = 4
	num_cores = 8 * num_slaves
	num_partitions = num_cores * 16
	conf = pyspark.SparkConf().setAppName("SparseFaceDetection")
	sc = pyspark.SparkContext(conf=conf)
	
	xml_file = "haarcascade_frontalface_default.xml"
	STAGES_BC = sc.broadcast(get_stages(xml_file))
	
	# split image into windows
	image_collection = ["family2.jpg", "Solvay.jpg", "Oscar.jpg", "Big_3.jpg", \
						"family1.jpg", "nasa.jpg", "Beatles.jpg"]
	windows = sc.parallelize(image_collection, \
							numSlices=len(image_collection))\
							.flatMap(img_2_windows)
	windows.persist()
	n_windows = windows.map(lambda e: len(e[1][0])).reduce(lambda a, b: a + b)
	size_group = (n_windows + num_partitions - 1) // num_partitions
	GROUP_SIZE_BC = sc.broadcast(size_group)
	print(f"\nn_windows = {n_windows}, size_group = {size_group}\n")
	
	# extract faces
	img_faces = windows.flatMap(group_windows).partitionBy(num_partitions)\
						.mapPartitions(filter_windows)\
						.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))\
						.mapValues(lambda v: img_nms(*v))\
						.collectAsMap()
	
	# save face as images
	s3 = boto3.resource('s3')
	for img in img_faces:
		pos, scores = img_faces[img]
		if len(pos) == 0:
			continue
		obj = s3.Object('image.collection', img)
		image = Image.open(obj.get()['Body'], 'r').convert('L')
		img_array = np.asarray(image).astype(int)
		for i, cur_pos in enumerate(pos):
			x, y, h, w = cur_pos
			face = img_array[int(x):int(x+h), int(y):int(y+w)].copy()
			im = Image.fromarray(face.astype('uint8'))
			im.save(f"output/{ img.split('.')[0] }_face_{i}.png")

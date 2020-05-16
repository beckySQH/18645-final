from lib import *
import cv2
import time

def run_my_version():

	xml_file_path = 'haarcascade_frontalface_default.xml'

	img_file_path = 'WechatIMG56.jpeg'
	images = ['WechatIMG56.jpeg','WechatIMG55.jpeg','WechatIMG57.jpeg','WechatIMG58.jpeg']
	print(time.perf_counter())
	for img in images:
		image = Image.open(img, 'r').convert('L')

		cascade = Cascade(xml_file_path, scale = 1.1, thresh = 0)

		faces = cascade.get_best_face(image)

		for face in faces:

			im = Image.fromarray(face.astype('uint8'))
			
			im.save(f"{ img_file_path.split('.')[0] }_face.png")

			plt.imshow(255 - face, 'Greys')

		# plt.show()
	



def run_reference():

	cascPath = 'haarcascade_frontalface_default.xml'
	imagePath = 'Beatles_small.jpg'

	faceCascade = cv2.CascadeClassifier(cascPath)

	gray = cv2.imread(imagePath, 0)

	# Detect faces
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		flags=cv2.CASCADE_SCALE_IMAGE
	)
	# For each face
	for (x, y, w, h) in faces: 
		# Draw rectangle around the face
		cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 3)

	plt.imshow(gray, cmap='gray')
	plt.show()


if __name__ == "__main__":
	print(time.perf_counter())
	run_my_version()
	print(time.perf_counter())

	# run_reference()
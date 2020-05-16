import numpy as np


"""Get the integral image.
		Integral Image:
		+ - - - - -        + -  -  -  -  -  -
		| 1 2 3 4 .        | 0  0  0  0  0  .
		| 5 6 7 8 .   =>   | 0  1  3  6 10  .
		| . . . . .        | 0  6 14 24 36  .
						   | .  .  .  .  .  .
		"""

def to_integral_image(img_arr):
	"""
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

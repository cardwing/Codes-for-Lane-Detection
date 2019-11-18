# Constants:
IN_IMAGE_W = 1640 # Input and final output image will be this size
IN_IMAGE_H = 590 # Input and final output image will be this size
VERTICAL_CROP_SIZE = 240 # We will crop this number of fixels from the image top
TRAIN_IMG_W = 976 # We will train model with this width
TRAIN_IMG_H = 208 # We will train model with this hights
POINTS_COUNT = 18 # Points count in the estimated curve
LANES_COUNT = 4 # Possible lanes count

IN_IMAGE_H_AFTER_CROP = IN_IMAGE_H - VERTICAL_CROP_SIZE # 350

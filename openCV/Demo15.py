import cv2
import numpy as np

# read the ImageNet class names
with open('../../input/classification_classes_ILSVRC2012.txt', 'r') as f:
   image_net_names = f.read().split('\n')
# final class names (just the first word of the many ImageNet names for one image)
class_names = [name.split(',')[0] for name in image_net_names]
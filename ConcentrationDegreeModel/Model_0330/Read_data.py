import tensorflow as tf
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
import os
import pickle

def read_data(image_dir, label_path):
    image_paths = np.sort(np.array(glob(image_dir)))
    images = np.array([np.asarray(Image.open(i)) for i in image_paths])
    labels = pd.read_csv(label_path, sep=',')
    labels = labels.sort_values(['image_id'])
    labels = labels['emotion'].to_numpy()
    return images, labels

train_image_dir = 'challengeA_data/images_train/*.jpg'
train_label_path = 'challengeA_data/challengeA_train.csv'

test_image_dir = 'challengeA_data/images_test/*.jpg'
test_label_path = 'challengeA_data/challengeA_test.csv'

train_images, train_labels = read_data(train_image_dir, train_label_path)
test_images, test_labels = read_data(test_image_dir, test_label_path)

print('train samples:', train_images.shape[0])
print('test samples:', test_images.shape[0])

def save_data(data_file, x_data, y_data):
    if not os.path.isfile(data_file):
        print('Saving data to pickle file...')
        try:
            with open(data_file, 'wb') as pfile:
                pickle.dump(
                    {'x_data': x_data,
                     'y_data': y_data},
                    pfile,
                    pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', data_file, ':', e)
            raise
    print('Data cached in pickle file.')

save_data('train_data.p', train_images, train_labels)
save_data('test_data.p', test_images, test_labels)
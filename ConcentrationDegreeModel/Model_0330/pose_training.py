import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import collections

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
import pandas as pd


def reshape_array(arr):
    # 将数组转换为 48x48 的二维数组
    new_arr = arr.reshape((48, 48))
    return new_arr
# def resize_face(face):
#     x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
#     return tf.image.resize(x, (48,48))
# def recognition_preprocessing(faces):
#     x = tf.convert_to_tensor([resize_face(f) for f in faces])
#     return x

def load_train_data(csv_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    # 从 DataFrame 中提取图像数据和标签数据
    x_data = df['pixels'].values
    y_data = df['pose'].values
    # 将图像数据转换为 NumPy 数组
    x_data = np.array([np.fromstring(x, dtype=int, sep=' ') for x in x_data])
    x_data = np.array([reshape_array(x) for x in x_data])
    # 将标签数据转换为 NumPy 数组
    y_data = np.asarray(y_data, dtype=np.int32)
    return x_data, y_data

# def load_data(data_file):
#     print('Loading data ...')
#     with open(data_file, 'rb') as f:
#         pickle_data = pickle.load(f)
#         x_data = pickle_data['x_data']
#         y_data = pickle_data['y_data']
#     print('Data loaded.')
#     return x_data, y_data

data_file = 'D:\\TrainData\\data\\data.csv'
images, labels = load_train_data(data_file)

n_samples = labels.shape[0]
print('Total samples:', n_samples)
print('images shape:', images.shape)
print('labels shape:', labels.shape)


#explor data
pose = {
    0: 'drink',
    1: 'listen',
    2: 'phone',
    3: 'trance',
    4: 'write'
}

num_classes = len(pose)


def plot_sample_distribution(labels):
    classes, cnts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(12, 5))
    plt.barh(list(pose.values()), cnts, height=0.6)
    for i, v in enumerate(cnts):
        plt.text(v, i, ' '+str(v), va='center')
    plt.xlabel('Counts')
    plt.title("Distribution of samples")

plot_sample_distribution(labels)

def show_images(images, labels, col=5):
    n = images.shape[0]
    row = np.ceil(n / col)
    fig = plt.figure(figsize=(2*col, 2*row))
    for i in range(n):
        fig.add_subplot(row, col, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(pose[labels[i]])
        plt.xticks([]), plt.yticks([])
    plt.show()

show_images(images[:25], labels[:25])

def show_one_emotion(images, labels, id, start=0, num=25):
    image_x = images[labels==id]
    label_x = labels[labels==id]
    show_images(image_x[start:start+num], label_x[start:start+num])

show_one_emotion(images, labels, id=1)

#split data
image_train, image_test, label_train, label_test = train_test_split(images, labels, test_size=0.2, random_state=42)
image_train, image_val, label_train, label_val = train_test_split(image_train, label_train, test_size=0.2, random_state=42)

print('Training samples:', label_train.shape[0])
print('Validation samples:', label_val.shape[0])
print('Test samples:', label_test.shape[0])

#向上采样训练数据
#SMOTE:可以有效地增加少数类样本的数量，提高分类模型的预测精度
def upsampling(x, y, strategy):
    (n, w, h) = x.shape
    sm = SMOTE(sampling_strategy=strategy, random_state=42)
    x_flat = x.reshape((n,-1))
    x_up, y_up = sm.fit_resample(x_flat, y)
    n_up = x_up.shape[0]
    x_up = x_up.reshape((n_up,w,h))

    return x_up, y_up

collections.Counter(label_train)
image_train_up, label_train_up = upsampling(image_train, label_train, 'auto')
collections.Counter(label_train_up)

for i in range(num_classes):
    if i == 3:
        continue
    show_one_emotion(image_train_up, label_train_up, id=i, start=4000, num=5)

#Utils
def one_hot_encoding(labels, num_classes):
    '''将每个标签映射到一个 num_classes 维的向量，其中第 i 个元素为 1 表示该样本属于第 i 类'''
    return tf.keras.utils.to_categorical(labels, num_classes)

def reshape_images(images, channel=1, resize=None):
    '''重塑图像尺寸和通道数的函数'''
    x = tf.expand_dims(tf.convert_to_tensor(images), axis=3)#最后一个维度上添加一个维度
    if channel > 1:
        x = tf.repeat(x, channel, axis=3)
    if resize is not None:
        x = tf.image.resize(x, resize)
    return x

def pre_processing(images, labels, num_classes, channel=1, resize=None, one_hot=True):
    '''将输入的图像和标签数据进行处理，以便于送入神经网络进行训练'''
    x = reshape_images(images, channel, resize)
    y = one_hot_encoding(labels, num_classes)
    return x, y


def plot_metrics(history):
    '''绘制训练过程中损失和准确率曲线的函数'''
    metrics = ['loss', 'accuracy']
    plt.figure(figsize=(15, 6))
    plt.rc('font', size=12)
    for n, metric in enumerate(metrics):
        name = metric.capitalize()
        plt.subplot(1,2,n+1)
        plt.plot(history.epoch, history.history[metric], label='Training', lw=3, color='navy')
        plt.plot(history.epoch, history.history['val_'+metric], lw=3, label='Validation', color='deeppink')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.title('Model '+name)
        plt.legend()
    plt.show()

def evaluate_class(model, x_test, y_test):
    '''评估分类模型的性能
    该函数首先使用 argmax 函数从 y_test 中提取真实标签，然后在每个情感标签下计算准确率并打印出来。
    最后，函数还打印了总体准确率
    '''
    labels = np.argmax(y_test, axis=1)
    print('{:<15}Accuracy'.format('Emotion'))
    print('-'*23)
    for i in range(num_classes):
        x = x_test[labels==i]
        y = y_test[labels==i]
        loss, acc = model.evaluate(x,  y, verbose=0)
        print('{:<15}{:.1%}'.format(pose[i], acc))
    print('-'*23)
    loss, acc = model.evaluate(x_test,  y_test, verbose=0)
    print('{:<15}{:.1%}'.format('Overall', acc))

def model_checkpoint_cb(file_path):
    '''用于在训练期间检查和保存具有最高验证准确度的模型权重'''
    return ModelCheckpoint(
        file_path, monitor='val_accuracy', mode='max',
        save_best_only=True, save_weights_only=True)

x_train, y_train = pre_processing(image_train_up, label_train_up, num_classes)
x_val, y_val = pre_processing(image_val, label_val, num_classes)
x_test, y_test = pre_processing(image_test, label_test, num_classes)

#定义了一个图像数据生成器对象train_datagen,用于对训练数据进行数据增强操作
'''rotation_range表示旋转范围,
shear_range表示剪切强度,
width_shift_range和height_shift_range表示平移范围,
zoom_range表示缩放范围,
horizontal_flip表示是否进行水平翻转'''
train_datagen = ImageDataGenerator(
    rotation_range=30,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True)


val_datagen = ImageDataGenerator()

batch_size = 128
#flow方法从训练和验证数据中动态地生成小批量数据
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(x_val, y_val)

steps_per_epoch = train_generator.n // train_generator.batch_size
input_shape = x_train[0].shape

class VGGNet(Sequential):
    def __init__(self, input_shape, num_classes, checkpoint_path, lr=1e-3):
        super().__init__()
        self.add(Rescaling(1./255, input_shape=input_shape))
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Flatten())
        
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(256, activation='relu'))

        self.add(Dense(num_classes, activation='softmax'))

        self.compile(optimizer=Adam(learning_rate=lr),
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])
        
        self.checkpoint_path = checkpoint_path

model = VGGNet(input_shape, num_classes, 'ConcentrationDegreeModel\\Model_0330\\saved_models\\fer_2013.h5')
model.summary()

epochs = 200
cp = model_checkpoint_cb(model.checkpoint_path)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-10)
es = EarlyStopping(monitor='val_loss', verbose=1, patience=20)

history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[lr, es, cp])

plot_metrics(history)
model.load_weights(model.checkpoint_path)
evaluate_class(model, x_test, y_test)
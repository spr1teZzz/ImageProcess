DATA_FILE = 'ConcentrationDegreeModel\Model_0304\\training_data\\'
NCLASSES  = 7
MODEL = 'ConcentrationDegreeModel\Model_0304\\models\\model_face.hdf5'
################################################################################
#
#                                LOADING DATA
#
################################################################################
import numpy as np
level = ['anger','contempt','disgust','fear','happy','sadness','surprise']
def loadData(classnumber):
    '''
        类加载训练数据函数
        =======
        PARAMS
        =======

         classnumber : int
              代表类的数字。

        =======
        RETURNS
        =======

         labels      : list of int
              A list containing the class of the elements loaded.包含所加载元素的类的列表。

         data        : list of floats
              A list containing the recorded data for that class.包含该类的记录数据的列表。
    '''
    print(f'[INFO] Loading data for class {level[classnumber]}')
    # Reading the file
    with open(DATA_FILE+level[classnumber]+'.txt', 'r') as f:
        bkgd = f.readlines()#读取数据的所有行
        f.close()#关闭文件

    # Removing unnecessary symbols 删除不必要的符号
    lines = [line.replace('[', '').replace(']','').replace('\n','').split(',') for line in bkgd]#获取每一行的数据

    # Getting the individual strings of just floats 获取单个浮点的字符串
    data = []
    labels = []
    label = np.zeros(NCLASSES)  #NCLASSES 10 生成10个元素0
    label[classnumber] = 1.0 #将下标 classnumber 的元素置1
    for  idx, line in enumerate(lines): #lines 分成 (0,lines[0]),(1,lines[1]),...
        data.append([])
        for val in line:
            data[idx].append(val)
        data[idx] = list(filter(None, data[idx]))#过滤掉为None的元素
        # Creates an array with class label 创建一个带有类标签的数组
        labels.append(label)

    # Converting them to floats  将它们转换为浮点数
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]=float(data[i][j])
    print(f'[INFO] Loaded {len(labels)} elements')
    # print('*************data**********************')
    # print(data)
    # print('***************labels********************')
    # print(labels)
    return data, labels

y = []
X = []
# Loading the data   加载数据
for i in range(NCLASSES):
    tX, ty = loadData(i)   #调用loadData函数 返回参数data, labels
    y.extend(ty)#添加新的列表内容ty
    X.extend(tX)#添加新的列表内容tX

# print(y)
# print(X)
################################################################################
#
#             将数据分成训练集和测试集
#
################################################################################
from sklearn.model_selection import train_test_split

RANDOM_SEED = 32

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state=RANDOM_SEED)
# print('###################################X_train#############################################')
# print(X_train)
# print('#####################################y_train###########################################')
# print(y_train)
# print('#####################################X_test###########################################')
# print(X_test)
# print('#####################################y_test###########################################')
# print(y_test)

################################################################################
#
#                           构建模型
#
################################################################################

import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((len(X_train[0]), )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(NCLASSES, activation='softmax')
])

# Show the summary of the model  显示模型的摘要
model.summary()

# What to save
cp_callback = tf.keras.callbacks.ModelCheckpoint( MODEL,
                                                  verbose=1,
                                                  save_weights_only=False
                                                 )
# Early stop
es_callback = tf.keras.callbacks.EarlyStopping(patience = 20, verbose=1)
model.compile( optimizer = 'adam',
               loss='categorical_crossentropy',
               metrics=['accuracy']
              )

################################################################################
#
#                             训练模型
#
################################################################################

model.fit( np.array(X_train),
           np.array(y_train),
           epochs=1000,
           batch_size=128,
           validation_data=(np.array(X_test),np.array(y_test)),
           callbacks=[cp_callback, es_callback]
           )
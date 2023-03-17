
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf 
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头
import pandas as pd #数据的读取
import urllib.request
import os
from sklearn import preprocessing
data_url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanice.xls"
data_file_path ="D:\\VScodeProject\\pythonProject\\TensorFlowDemo\\Demo\\KerasApplication\\titanic3.xls"

if not os.path.isfile(data_file_path):
    result = urllib.request.urlretrieve(data_url,data_file_path)
    print("downloaded:",result)
else:
    print(data_file_path,"data file already exits.")


#读取数据文件，结果为DataFrame格式
df_data = pd.read_excel(data_file_path)
selected_cols = ["survived","name","pclass","sex","age","sibsp","parch","fare","embarked"]
selected_df_data = df_data[selected_cols]
def prepare_data(df_data):
    #删除name字段
    #drop不改变原有的df中的数据，而是返回另一个DataFrame来存放删除后的数据axis =1 表示删除列
    df = df_data.drop(["name"],axis=1)

    #为缺失age记录填充 设置为平均值
    age_mean_value = df["age"].mean()
    df["age"] = df["age"].fillna(age_mean_value)
    #为确实fare记录填充值
    fare_mean_value = df["fare"].mean()
    df["fare"] = df["fare"].fillna(fare_mean_value)
    #为确实embarked记录填充值
    df["embarked"] = df["embarked"].fillna("S")
    #转换编码
    #性别sex由字符串转为数字编码
    df["sex"] = df["sex"].map({"female":0,"male":1}).astype(int)
    #港口embarked由字母表示转换为数字编码
    df["embarked"] = df["embarked"].map({"C":0,"Q":1,"S":2}).astype(int)

    ndarray_data = df.values
    #后7列是特征值
    features = ndarray_data[:,1:]
    #第0列是标签值
    label = ndarray_data[:,0]
    #特征值标准化
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
    norm_features = minmax_scale.fit_transform(features)
    return norm_features,label

#打乱数据,为后面训练做准备
shuffled_df_data = selected_df_data.sample(frac=1)
#处理数据
x_data,y_data = prepare_data(shuffled_df_data)

#划分训练集和测试集
train_size = int(len(x_data)*0.8)

x_train = x_data[:train_size]
y_train = y_data[:train_size]

x_test = x_data[train_size:]
y_test = y_data[train_size:]


#建立Keras序列模型
model = tf.keras.models.Sequential()
#加入第一层,输入特征数据是7列,也可以用input_shape=(7,)
model.add(tf.keras.layers.Dense(units=64,
                                input_dim=7,
                                use_bias=True,#使用偏置项
                                kernel_initializer="uniform",#权重初始化方式
                                bias_initializer="zeros",
                                activation="relu"))

#model.add(tf.keras.layers.Dropout(rate=0.3))#防止过拟合
model.add(tf.keras.layers.Dense(units=32,
                                activation="sigmoid"))#激活函数

model.add(tf.keras.layers.Dense(units=1,
                                activation="sigmoid"))
#模型摘要
print(model.summary())
model.compile(optimizer=tf.keras.optimizers.Adam(0.003),#优化器
            #损失函数名 sigmoid作为激活函数，一般选择binary_crossentropy
            #softmax作为激活函数,一般损失函数选用categorical_crossentropy
            loss="binary_crossentropy",
            metrics=["accuracy"])#模型要训练和评估的度量值
#设置回调参数,内置的回调还包括:
#tf.keras.callbacks.LearningRateScheduler() #动态调整学习率
#tf.keras.callbacks.EarlyStopping #自动判断 早停
logdir = "TensorFlowDemo\\Demo\\KerasApplication\\logs"
checkpoint_path = "TensorFlowDemo\\Demo\\KerasApplication\\checkpoint\\Titanic.{epoch:02d}-{val_loss:.2f}.ckpt"

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                            histogram_freq=2),#频率
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,#检查点文件值保存模型的权值
                                                verbose=1,#是否显示
                                                period=5)]#保存周期


#模型训练
train_history = model.fit(x=x_train,y=y_train,validation_split=0.2,epochs=100,batch_size=40,callbacks=callbacks,verbose=2)

def visu_train_history(train_history,train_metric,validation_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[validation_metric])
    plt.title("Train History")
    plt.ylabel(train_metric)
    plt.xlabel("epoch")
    plt.legend(["train","validation"],loc="upper left")
    plt.show()

# print(train_history.history)
print(train_history.history.keys())

# visu_train_history(train_history,'accuracy','val_accuracy')

evaluate_result = model.evaluate(x=x_test,y=y_test)
print(evaluate_result)
print(model.metrics_names)

#进行预测
Jack_info = [0,"Jack",3,"male",23,1,0,5.0000,"S"]
Rose_info = [1,"Rose",1,"female",20,1,0,100.0000,"S"]
#创建新的旅客DataFrame
new_passenger_pd = pd.DataFrame([Jack_info,Rose_info],columns=selected_cols)
# #在老的DataFrame中加入新的旅客信息
all_passenger_pd = selected_df_data.append(new_passenger_pd)
#数据准备
x_features,y_label = prepare_data(all_passenger_pd)
#利用模型计算旅客生存概率
surv_probability = model.predict(x_features)
#在数据表最后一列插入生存概率
all_passenger_pd.insert(len(all_passenger_pd.columns),"surv_probability",surv_probability)
print(all_passenger_pd[-5:])


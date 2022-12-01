#波士顿房价预测
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf 
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头
import pandas as pd #数据的读取
from sklearn.utils import shuffle

#读取数据
df = pd.read_csv("TensorFlowDemo\\data\\boston.csv",header=0)
# print(df.describe())

#获取df的值
df = df.values
#把df转换为np的数组格式
df  =np.array(df)

#对特征数据[0-11]列做(0-1)归一化
for i in range(12):
    df[:,i] = df[:,i] / (df[:,i].max()-df[:,i].min())

#x_data  为前12列特征数据
x_data = df[:,:12]
#y_data 为最后一列标签数据
y_data = df[:,12]
# print(x_data,'\n',x_data.shape)
# print(y_data,'\n',y_data.shape)

x = tf.placeholder(tf.float32,[None,12],name="X") #12个特征数据(12列)
y = tf.placeholder(tf.float32,[None,1],name="Y") #1个标签数据(1列)

#定义了一个命名空间
with tf.name_scope("Model"):
    #w 初始化为shape = (12,1)的随机数
    w = tf.Variable(tf.random_normal([12,1],stddev=0.01),name="W")

    #b 初始化值为 1.0
    b = tf.Variable(1.0,name="b")

    #w和x是矩阵相乘，用matmul，不能用mutiply 或者*
    def model(x,w,b):
        return tf.matmul(x,w)+b

    #预测计算操作，向前计算节点
    pred = model(x,w,b)

    #迭代次数
    train_epochs = 100
    #学习率
    learning_rate = 0.01

    #定义损失函数
    with tf.name_scope("LossFunction"):
        loss_function = tf.reduce_mean(tf.pow(y-pred,2))#均方误差

    #创建优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

    #声明会话
    sess = tf.Session()
    #定义初始化变量的操作
    init = tf.global_variables_initializer()
    #启动回话
    sess.run(init)

    loss_list = [] #用于保存loss值的列表
    #迭代训练
    for epoch in range(train_epochs):
        loss_sum = 0.0
        for xs,ys in zip(x_data,y_data):
            #数据变形,要和Placeholder的shape一致
            xs = xs.reshape(1,12)
            ys = ys.reshape(1,1)

            _,loss = sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
            #计算本轮loss值的和
            loss_sum = loss_sum+loss
            loss_list.append(loss)
        #打乱数据顺序
        xvalues,yvalues = shuffle(x_data,y_data)

        b0temp = b.eval(session = sess)
        w0temp = w.eval(session = sess)
        loss_average = loss_sum /len(y_data)
        #print("epoch=",epoch+1,"loss=",loss_average,"b=",b0temp,"w=",w0temp)
        #loss_list.append(loss_average)

    plt.plot(loss_list)
    plt.show()
    #特定数据
    n = 348
    x_test = x_data[n]
    x_test = x_test.reshape(1,12)
    predict =sess.run(pred,feed_dict = {x:x_test})
    print("预测值:%f"%predict)
    target = y_data[n]
    print("标签值:%f"%target)

    #随机数据
    n = np.random.randint(506)
    x_test = x_data[n]
    x_test = x_test.reshape(1,12)
    predict =sess.run(pred,feed_dict = {x:x_test})
    print("预测值:%f"%predict)
    target = y_data[n]
    print("标签值:%f"%target)


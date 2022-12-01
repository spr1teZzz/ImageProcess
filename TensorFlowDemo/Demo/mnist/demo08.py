#从models中还原模型
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf 
import os
from time import time
import tensorflow.examples.tutorials.mnist.input_data as input_data
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头

mnist = input_data.read_data_sets("TensorFlowDemo\\data\\MNIST_data\\",one_hot = True)
#定义全连接层函数
def fcn_layer(inputs,input_dim,output_dim,activation=None):
    #输入神经元 输出神经元 输出神经元数量 激活函数
    W = tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev=0.1))#以截取正态分布的随机数初始化W
    b = tf.Variable(tf.zeros([output_dim]))#以0初始化b

    XWb = tf.matmul(inputs,W) +b #建立表达式:inputs * W +b

    if activation is None:#默认不使用激活函数
        outputs = XWb
    else:#若传入激活函数,则用其对输出结果进行变换
        outputs = activation(XWb)
    return outputs

#单层--------------------------------------------------------------------------
# #构建输出层
# x  = tf.placeholder(tf.float32,[None,784],name="X")
# #0-9 一共十个数字--->10个类别
# y = tf.placeholder(tf.float32,[None,10],name="Y")
# #隐藏层
# h1 =fcn_layer(inputs=x,input_dim=784,output_dim=256,activation=tf.nn.relu)
# #输出层
# forward = fcn_layer(inputs=h1,input_dim=256,output_dim=10,activation=None)
# pred = tf.nn.softmax(forward)
#单层--------------------------------------------------------------------------

#多层--------------------------------------------------------------------------
x = tf.placeholder(tf.float32,[None,784],name="X")
y = tf.placeholder(tf.float32,[None,10],name="Y")
H1_NN = 256
H2_NN = 64

#构建隐藏层1
h1 = fcn_layer(inputs=x,input_dim=784,output_dim=H1_NN,activation=tf.nn.relu)
#构建隐藏层2
h2 = fcn_layer(inputs=h1,input_dim=H1_NN,output_dim=H2_NN,activation=tf.nn.relu)
#构建输出层
forward = fcn_layer(inputs=h2,input_dim=H2_NN,output_dim=10,activation=None)
pred = tf.nn.softmax(forward)
#多层--------------------------------------------------------------------------

#定义准确率
#检查预测类别tf.argmax(pred,1) 与实际类别 tf.argmax(y,1)的匹配情况
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
#准确率,将布尔值转化为浮点数，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#创建保存模型文件的目录
ckpt_dir = "D:\\VScodeProject\\pythonProject\\TensorFlowDemo\\models"

#创建saver
saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

ckpt = tf.train.get_checkpoint_state(ckpt_dir)

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess,ckpt.model_checkpoint_path)#从已保存的模型中读取参数
    print("Restore model from "+ckpt.model_checkpoint_path)

print("Accuracy:",accuracy.eval(session = sess,feed_dict = {x:mnist.test.images,y:mnist.test.labels}))

#评估模型
accu_train = sess.run(accuracy,feed_dict = {x:mnist.train.images,y:mnist.train.labels})
print("Test Accuracy:",accu_train) 
#进行预测
prediction_result = sess.run(tf.argmax(pred,1),feed_dict={x:mnist.test.images})
def print_predict_errs(labels,prediction):
    #标签列表,预测值列表
    count = 0
    compare_lists = (prediction == np.argmax(labels,1))
    err_lists = [i for i in range(len(compare_lists)) if compare_lists[i] == False]
    for x in err_lists:
        print("index="+str(x)+"标签值=",np.argmax(labels[x]),"预测值=",prediction[x])
        count = count+1
    print("总计:"+str(count))
print_predict_errs(labels=mnist.test.labels,prediction=prediction_result)

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


# #定义交叉熵损失函数
# loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices = 1))

#TensorFlow 提供了softmax_cross_entropy_with_logits函数
#用于避免因为log(0)值为NaN造成的数据不稳定
loss_function =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = forward,labels = y))


#设置训练参数
train_epochs = 40 #训练轮数
batch_size = 50 #单次训练样本数(批次大小)
total_batch = int(mnist.train.num_examples/batch_size)#一轮训练有多少批次
display_step = 1 #显示粒度
learning_rate = 0.01 #学习率

#选择优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
#定义准确率
#检查预测类别tf.argmax(pred,1) 与实际类别 tf.argmax(y,1)的匹配情况
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
#准确率,将布尔值转化为浮点数，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#保存模型
#存储模型的粒度
save_step = 5
#创建保存模型文件的目录
ckpt_dir = "D:\\VScodeProject\\pythonProject\\TensorFlowDemo\\models"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

#声明完所有变量后,调用tf.train.Saver
saver = tf.train.Saver()
startTime = time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs,ys = mnist.train.next_batch(batch_size) #读取批次数据
        sess.run(optimizer,feed_dict = {x:xs,y:ys})#执行批次训练

    #total_batch个批次训练完成后,使用验证数据计算误差与准确率
    loss,acc = sess.run([loss_function,accuracy],feed_dict = {x:mnist.validation.images,y:mnist.validation.labels})

    if(epoch+1)%display_step == 0:
        print("Train Epoch:","%02d"%(epoch+1),"Loss=","{:.9f}".format(loss),"Accuracy=","{:.4f}".format(acc))

    if(epoch+1)%save_step == 0:
        saver.save(sess,os.path.join(ckpt_dir,'mnist_h256_model_{:06d}.ckpt'.format(epoch+1)))#存储模型
saver.save(sess,os.path.join(ckpt_dir,'mnist_h256_model.ckpt'))
#显示运行总时间
duration = time() - startTime
print("Train Finished takes:","{:.2f}".format(duration))

#评估模型
accu_train = sess.run(accuracy,feed_dict = {x:mnist.train.images,y:mnist.train.labels})
print("Test Accuracy:",accu_train) 







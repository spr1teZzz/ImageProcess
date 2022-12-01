import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头

mnist = input_data.read_data_sets("TensorFlowDemo\\data\\MNIST_data\\",one_hot = True)
#构建模型
#mnist 中每张图片共有28x28 = 784个像素点
x = tf.placeholder(tf.float32,[None,784],name="X")
#0-9 一共十个数字--->10个类别
y = tf.placeholder(tf.float32,[None,10],name="Y")


H1_NN = 256 #第一隐藏神经元为256个
H2_NN = 64  #第二隐藏神经元为64个

#隐藏层神经元数量
#输入层-第1隐藏参数和偏置项
W1 = tf.Variable(tf.truncated_normal([784,H1_NN],stddev=0.1))
b1 = tf.Variable(tf.zeros([H1_NN]))

#第1隐藏层 - 第2隐藏层参数和偏置项
W2= tf.Variable(tf.truncated_normal([H1_NN,H2_NN],stddev=0.1))
b2 = tf.Variable(tf.zeros([H2_NN]))

#第2隐藏层-输出层参数和偏置项 
W3= tf.Variable(tf.truncated_normal([H2_NN,10],stddev = 0.1))
b3 = tf.Variable(tf.zeros([10]))

#计算第1隐藏层结果
Y1 = tf.nn.relu(tf.matmul(x,W1)+b1)
#计算第2隐藏层结果
Y2 = tf.nn.relu(tf.matmul(Y1,W2)+b2)
#计算输出结果
forward = tf.matmul(Y2,W3)+b3
pred = tf.nn.softmax(forward)

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
from time import time
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
#显示运行总时间
duration = time() - startTime
print("Train Finished takes:","{:.2f}".format(duration))

#评估模型
accu_train = sess.run(accuracy,feed_dict = {x:mnist.train.images,y:mnist.train.labels})
print("Test Accuracy:",accu_train)

#进行预测
prediction_result = sess.run(tf.argmax(pred,1),feed_dict={x:mnist.test.images})
#查看预测结果中的前10项
prediction_result[0:10]

# compare_lists = prediction_result == np.argmax(mnist.test.labels,1)
# print(compare_lists)

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

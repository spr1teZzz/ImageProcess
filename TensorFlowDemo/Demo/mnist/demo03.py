import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头



mnist = input_data.read_data_sets("TensorFlowDemo\\data\\MNIST_data\\",one_hot = True)
# print("训练集 train 数量:",mnist.train.num_examples,
#     ",验证集 validation 数量:",mnist.validation.num_examples,
#     ",测试集test数量:",mnist.test.num_examples)

# print("train images shape:",mnist.train.images.shape,"labels shapl e:",mnist.train.labels.shape)



# def polt_image(image):
#     plt.imshow(image.reshape(28,28),cmap="binary")
#     plt.show()
# polt_image(mnist.train.images[0])
# polt_image(mnist.train.images[1000])

# print(mnist.train.labels[1])
# np.argmax(mnist.train.labels[1])

#mnist_no_one_hot = input_data.read_data_sets("TensorFlowDemo\\data\\MNIST_data\\",one_hot = False)
# print(mnist_no_one_hot.train.labels[0:10])


# print("validation images:",mnist.validation.images.shape,"labels:",mnist.validation.labels.shape)
# print("test images:",mnist.test.images.shape,"labels:mnist.test.labels.shape")

# # print(mnist.train.labels[0:10])
# batch_images_xs,batch_images_ys = mnist.train.next_batch(batch_size = 10)
# print(batch_images_xs.shape,batch_images_ys.shape)
# print(batch_images_ys)
# print(batch_images_xs)


#测试Softmax
# x = np.array([[-3.1,1.8,9.7,-2.5]])
# pred = tf.nn.softmax(x)#Softmax分类
# sess = tf.Session()#声明会话
# v = sess.run(pred)
# print(v)
# sess.close()


#构建模型
#mnist 中每张图片共有28x28 = 784个像素点
x = tf.placeholder(tf.float32,[None,784],name="X")
#0-9 一共十个数字--->10个类别
y = tf.placeholder(tf.float32,[None,10],name="Y")

#创建变量
W = tf.Variable(tf.random_normal([784,10]),name="W")
b = tf.Variable(tf.zeros([10]),name="b")

#用单个神经元构建神经网络
forward = tf.matmul(x,W)+b #前向计算
pred = tf.nn.softmax(forward) #Softmax分类
#设置训练参数
train_epochs = 50 #训练轮数
batch_size = 100 #单次训练样本数(批次大小)
total_batch = int(mnist.train.num_examples/batch_size)#一轮训练有多少批次
display_step = 1 #显示粒度
learning_rate = 0.01 #学习率

#定义损失函数
#定义交叉熵损失函数
loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices = 1))

#选择优化器
#梯度优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

#定义准确率
#检查预测类别tf.argmax(pred,1) 与实际类别 tf.argmax(y,1)的匹配情况
correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

#准确率,将布尔值转化为浮点数，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#声明会话
sess = tf.Session()
init = tf.global_variables_initializer()#变量初始化
sess.run(init)

#开始训练
for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs,ys = mnist.train.next_batch(batch_size)#读取批次数据
        sess.run(optimizer,feed_dict={x:xs,y:ys})#执行批次训练
    #total_batch 个批次训练完成之后，使用验证数据计算误差与准确率；验证集没有分批
    loss,acc = sess.run([loss_function,accuracy],feed_dict = {x:mnist.validation.images,y:mnist.validation.labels})

    #打印训练过程中的详细信息
    if(epoch+1)%display_step == 0:
        print("Train Epoch:","%02d"%(epoch+1),"Loss=","{:.9f}".format(loss),"Accuracy=","{:.4f}".format(acc))
        
print("Train Finished")
accu_validation = sess.run(accuracy,feed_dict ={x:mnist.validation.images,y:mnist.validation.labels})
print("Test Accuracy:",accu_validation)
accu_train = sess.run(accuracy,feed_dict = {x:mnist.train.images,y:mnist.train.labels})
print("Test Accuracy:",accu_train)


prediction_result = sess.run(tf.argmax(pred,1),feed_dict = {x:mnist.test.images})
# print(prediction_result[0:10])

#可视化操作
def plot_images_labels_prediction(images,labels,prediction,index,num=10):
    #图像列表,标签列表,预测值列表,从第index个开始显示,缺省一次显示10幅
    fig = plt.gcf() #获取当前图表,Get Current Figure
    fig.set_size_inches(10,12) #1英寸等于2.54cm
    if num>25:
        num = 25 #最多显示25个子图
    for i in range(0,num):
        ax = plt.subplot(5,5,i+1)#获取当前要处理的子图
        ax.imshow(np.reshape(images[index],(28,28)),cmap="binary")#显示第index个图像
        title = "label="+str(np.argmax(labels[index]))#构建该图上要显示的title
        if len(prediction)>0:
            title += ",predict="+str(prediction[index])
        ax.set_title(title,fontsize=10) #显示图上的title信息
        ax.set_xticks([])#不显示坐标轴
        ax.set_yticks([])
        index+=1
    plt.show()
plot_images_labels_prediction(mnist.test.images,mnist.test.labels,prediction_result,10,25)
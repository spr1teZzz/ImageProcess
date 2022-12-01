#人工数据集生成

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头

#设置随机数种子
np.random.seed(5)

#直接采用np生成等差数列的方法，生成100个点，每个点的取值在-1-1之间
x_data = np.linspace(-1,1,100)

#y = 2X +1 +噪声 ,其中，噪声的维度与x_data一致
y_data = 2 * x_data +1.0 +np.random.randn(*x_data.shape) *0.4

# # print(np.random.randn(10))
# plt.scatter(x_data,y_data)

# plt.plot(x_data,2*x_data+1.0,color="red",linewidth =3)
# plt.show()

x = tf.placeholder("float",name="x")
y = tf.placeholder("float",name="y")

#构建回归模型
def model(x,w,b):
    return tf.multiply(x,w)+b

#构建线性函数的斜率，变量w
w = tf.Variable(1.0,name="w0")
#构建线性函数的截距,变量b
b = tf.Variable(0.0,name="b0")
#pred是预测值，向前计算
pred = model(x,w,b)

#迭代次数(训练轮数)
train_epochs = 10
#学习率
learning_rate = 0.05
#采用均方差作为损失函数
loss_function = tf.reduce_mean(tf.square(y-pred))
#梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

step = 0 #记录训练步数
loss_list = [] #用于保存loss值的列表
display_step = 10
for epoch in range(train_epochs):
    for xs,ys in zip(x_data,y_data):
        _,loss=sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})

        loss_list.append(loss)
        step = step+1
        if step % display_step ==0:
            print("Train Epoch:","%02d" % (epoch+1),"Step:%03d"%(step),"loss=","{:.9f}".format(loss))
    b0temp = b.eval(session = sess)
    w0temp = w.eval(session = sess)
    #plt.plot(x_data,w0temp*x_data+b0temp)#画图

# plt.show()
# print("w:",sess.run(w)) #w的值应该在2附近
# print("b:",sess.run(b)) #b的值应该在1附近

# #可视化
# plt.scatter(x_data,y_data,label="Original data")
# plt.plot(x_data,x_data*sess.run(w) + sess.run(b),label="Fitted line",color="r",linewidth = 3)
# plt.show()

plt.plot(loss_list)
plt.plot(loss_list,"r+")
plt.show()
# #预测
# x_test = 3.21
# predict = sess.run(pred,feed_dict={x:x_test})
# print("预测值:%f"%predict)
# target = 2*x_test +1.0
# print("目标值:%f"%target)

# x_test = 3.21
# predict = sess.run(w)*x_test+sess.run(b)
# print("预测值：%f"%predict)
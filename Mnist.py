#准备数据集
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("Mnist_data/",one_hot=True)

#建立模型  softmax
x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)#预测值

#建立模型指标  交叉熵
y_=tf.placeholder(tf.float32,[None,10])#真实值
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
train_stap=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#启动图,初始化所有变量
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#训练模型
for i in range(1000):#训练1000次
    batch_xs,batch_ys=mnist.train.next_batch(100)#按批次训练，每一批样本容量100
    sess.run(train_stap,feed_dict={x: batch_xs, y_: batch_ys})#feed_dict字典填充，将batch_x和batch_ys分别填充到x,y_这两个占位符中

#模型评估
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))#比较预测的标签值和真实标签是否相等，是返回True
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))#reduce_mean取均值，tf.cast将布尔类型转化成float
#输出结果
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
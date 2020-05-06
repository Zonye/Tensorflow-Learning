
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("Mnist_data/",one_hot=True)

''' 
tf.InteractiveSession()是一种交互式的session方式，它让自己成为了默认的session，
也就是说用户在不需要指明用哪个session运行的情况下，就可以运行起来.
run()和eval()函数可以不指明session
'''
sess = tf.InteractiveSession()

#定义权值函数
def weight_variable(shape):
    #截断的产生正态分布的随机数，即随机数与均值（0）的差值若大于两倍的标准差（0.1），则重新生成。
    initial=tf.truncated_normal(shape,stddev=0.1) #将权值设为[-0.2,0.2]之间的服从正太分布的随机数
    return tf.Variable(initial)

#定义偏置函数
def bias_variable(shape):
    #constant：第一个参数value是数字时，张量的所有元素都会用该数字填充，会生成一个shape大小的张量。
    initial=tf.constant(0.1,shape=shape)#偏置的值设为0.1。
    return tf.Variable(initial)

#卷积
def conv2d(x,W):
    '''
    tf.nn.conv2d (input, filter, strides, padding, use_ cudnn on_ gpu=None,name=None)
    input:输入图像X,它的形状是4维张量[batch,height,width,channel]
    file:过滤器W，形状【长，宽，通道数，过滤器个数】
    过滤器W在图像X上做[batch,height,width,channel]步长都为1的卷积操作。
    same:p为每个边缘填充层数，f为过滤器大小。p=（f-1)/2.当same卷积时，只有在步长为1的情况下，输出和输入大小相等
    输出大小（n+2p-f+1)
    '''
    return  tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

#池化
def max_pool_2x2(x):
  # tf.nn.max_pool(value, ksize, strid
  # es, padding, name=None)
   #ksize：池化窗口的大小，参数为四维向量，通常取[1, height, width, 1],
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],#池化模板大小为2x2
                        strides=[1, 2, 2, 1], padding='SAME')#步长为2,输出大小【（n+2p-f)/s】+1


x=tf.placeholder(tf.float32,[None,784])

x_image = tf.reshape(x, [-1,28,28,1])#将图像形状变成和过滤器的一样

#第一层卷积
W_conv1=weight_variable([5,5,1,32]) #第一层卷积设置32个5x5x1的过滤器
b_conv1=bias_variable([32])

h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#28x28x32，输出=n+2p-f+1
h_pool1=max_pool_2x2(h_conv1)#14x14x32

#第二层卷积
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#14x14x64
h_pool2=max_pool_2x2(h_conv2)#7x7x64

#全连接层
W_fc1=weight_variable([7*7*64,1024])#将权值设为一个7*7*64行，1024列的矩阵
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])#把最后一层池化层变成1行7*7*64列的二维张量
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#dropout，防止过拟合
keep_prob=tf.placeholder("float")

tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
#x：输入数据
#keep_prob: 表示的是保留的比例，假设为0.8 则 20% 的数据变为0，然后其他的数据乘以 1/keep_prob；keep_prob 越大，保留的越多
#用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率,

h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#softmax输出层
W_fc2=weight_variable([1024,10])#1024x10矩阵，因为最后输出结果是10类
b_fc2=bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)#[1,1024]x[1024,10],输出预测值

#优化模型
y=tf.placeholder(tf.float32,[None,10]) #真实值
cross_entropy=-tf.reduce_sum(y*tf.log(y_conv))#交叉熵损失函数做指标
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#Adam优化器使损失值最小

#判断模型准确率
correct_predict=tf.equal(tf.argmax(y,1),tf.argmax(y_conv,1))#tf.argmax函数给出某个tensor对象在某一维上的其数据最大值所在的索引值。
accuracy=tf.reduce_mean(tf.cast(correct_predict,"float"))#tf.cast将上面equal返回的布尔类型转化成float,然后取均值

#启动图

tf.global_variables_initializer().run()

#训练模型
for i in range(20000):
    batch=mnist.train.next_batch(50)#50一批
    if i%100==0:#每迭代100次，打印训练准确率
        #激发tensor.eval()这个函数之前，tensor的图必须已经投入到session里面，或者一个默认的session是有效的，或者显式指定session.
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})#batch[0],batch[1],0和1????
        print( "step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

    print( "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))







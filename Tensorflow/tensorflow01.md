## TensorFlow的计算图       

tensorflow在执行计算时需要使用计算图，图中的每个节点一般用来表示施加的数学操作，但是也可以表示数据输入的起点和输出的终点，或者是读取/写入持久变量的终点      

边表示节点之间的输入/输出关系，这些数据边可以传送维度可以动态调整的多维数据数组，即tensor。

tf会默认维护一个计算图，通过`tf.get_default_graph()`函数获取当前默认计算图    

```Python
# 通过a.graph可以查看张量所属的计算图
print(a.graph is tf.get_default_graph()) 
# 没有特意指定a的计算图，所以返回True
```      

### 生成新的计算图     

```Python
g1 = tf.Graph()
with g1.as_default():
    # 在计算图g1中定义变量v,并设置初始化为0
    v = tf.get_variable(
        "v", initializer = tf.zeros_initializer(shape=[1])
    )

g2 = tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量v，并设置初始化为1
    v = tf.get_variable(
        "v",initializer = tf.ones_initializer(shape=[1])
    )

# 在计算图g1中读取变量v的值
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variable().run()
    with tf.variable_scope("", reuse = True):
        # 在计算图g1中，变量v取值应该为0，所以下面这行会输出[0.]
        print(sess.run(tf.get_variable("v")))

# 在计算图g2中读取变量v的值
with tf.Session(graph=g2) as sess:
    tf.initialize_all_variable().run()
    with tf.variable_scope("", reuse = True):
        # 在计算图g2中，变量v取值应该为1，‘
        print(sess.run(tf.get_variable("v")))
```

上面的代码产生了两个计算图，每个计算图中都定义了一个v变量，但是当运行到不同的计算图中的v时，他们的值也是不一样的，tf中的计算图不仅仅可以**隔离张量和计算**，还提供了管理张量和计算的机制           


### 指定运算设备   

```Python
g = tf.Graph()
# 指定运算设备
with g.device('/gpu:0'):
    result = a + b
```


### sigmoid_cross_entropy_with_logits      

`def sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)`

* 计算公式:    

    `max(x,0) - x * z + log(1 + exp(-abs(x)))`     

    **logits**和**targets**必须有相同的数据维度和类型     


* 推导过程：    

    ```
    z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))
    ```  

### tf.trainable_variables      

该方法返回当前计算图中所有没有被标记`trainable=False`的变量集合     

### tf.global_variables_initalizer     

该方法将所有全局变量的初始化器汇总。    

其实该OP是一个`NoOp`,即不存在输入和输出，所有变量的初始化器都是通过控制依赖边与该`NoOp`相连，保证所有全局变量都被初始化     

### tf的占位符和feed_dict   

占位符没有初始值，只会分配必要的内存，在`Session`中，占位符可以使用`feed_dict`递送数据。     

在训练神经网络时每次需要提供一个批量的训练样本，如果每次迭代的数据都用常量来表示，那么最终计算图会非常的大，因为每增加一个常量，就要在计算图中增加一个节点，如果使用占位符，在计算图中只会有一个节点，而且我们可以根据批次的更新给这个节点递送不同的训练数据。       

```python
w1 = tf.Variable(tf.random_normal([1,2], stddev=1, seed=1))

# 因为需要重复输入x，每生成一个x就会增加一个节点，所以这里使用占位符
x = tf.placeholder(tf.float32, shape=(1,2))
x1 = tf.constant([[0.7, 0.9]])

a = x + w1
b = x1 + w1

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 运行y时给占位符喂值，feed_dict为字典，变量名不可变
y_1 = sess.run(a, feed_dict={x: [[0.7, 0.9]]})
y_2 = sess.run(b)

print(y_1)
print(y_2)
sess.close()
```  

占位符使用实例：     

```python
import tensorflow as tf
import numpy as np

list_of_points1_ = [[1, 2], [3, 4], [5, 6], [7, 8]]
list_of_points2_ = [[15, 16], [13, 14], [11, 12], [9, 10]]

list_of_points1 = np.array([np.array(elem).reshape(1, 2) for elem in list_of_points1_])
list_of_points2 = np.array([np.array(elem).reshape(1, 2) for elem in list_of_points2_])
print(list_of_points1)
print(list_of_points2)

graph = tf.Graph()

with graph.as_default():
    # 创建placeholder
    point1 = tf.placeholder(tf.float32, shape=(1, 2))
    point2 = tf.placeholder(tf.float32, shape=(1, 2))


    def calculate_eucledian_distance(point1, point2):
        difference = tf.subtract(point1, point2)
        power2 = tf.pow(difference, tf.constant(2.0, shape=(1, 2)))
        add = tf.reduce_sum(power2)
        eucledian_distance = tf.sqrt(add)
        return eucledian_distance


    dist = calculate_eucledian_distance(point1, point2)

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for ii in range(len(list_of_points1)):
        point1_ = list_of_points1[ii]
        point2_ = list_of_points2[ii]

        feed_dict = {point1: point1_, point2: point2_}
        distance = sess.run([dist], feed_dict=feed_dict)
        print("the distance between {} and {} -> {}".format(point1_, point2_, distance))

```   

输出：    

```
the distance between [[1 2]] and [[15 16]] -> [19.79899]
the distance between [[3 4]] and [[13 14]] -> [14.142135]
the distance between [[5 6]] and [[11 12]] -> [8.485281]
the distance between [[7 8]] and [[ 9 10]] -> [2.828427]
```






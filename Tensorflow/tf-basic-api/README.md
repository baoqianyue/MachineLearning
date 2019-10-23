## Tensorflow基础API   

### [Tensorflow数据类型](./tf-basic-api.ipynb)   
* tf.constant   
    定义tensor常量   
* tensor.numpy()     
    tensor类型转换为numpy类型   
* tf.strings    
    * tf.strings.length   
        输出strings长度，如果strings是字符串数组，会输出每个字符串的长度    
    * tf.strings.unicode_decode   
        对strings进行unicode编码   
* tf.ragged.constant    
    定义一个ragged tensor(不完整tensor)    
* tf.ragged.tensor.to_tensor()    
    将一个ragged tensor转换为tensor，空缺位置补0   
* tf.Variable    
    定义变量    
* variable.value()   
    将一个变量转换为tensor常量     
* variable.assign()     
    对一个变量进行赋值，可以选定位置进行赋值     

### [使用子类继承方式自定义layer](./tf_keras_regression_customized_layer.ipynb)   

```python
class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        # 该层的单元数，即output_shape
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """初始化所需参数"""
        # x * w + b 
        # w的参数确定，input_shape:[None, a] w:[a, b] output_shape :[None, b]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)
        
    def call(self, x):
        """完成正向传播"""
        return self.activation(x @ self.kernel + self.bias)   

# 模型构建
model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu',
                      input_shape=X_train.shape[1:]),
    CustomizedDenseLayer(1),
    customized_softplus,
    # 最后一层等同于：
    # keras.layers.Dense(1, activation='softplus')
    # keras.layers.Dense(1), keras.layers.Activation('softplus')
])

model.summary()
model.compile(loss='mean_squared_error', optimizer='sgd')
``` 

### [tf.function的使用](./tf_function_and_auto_graph.ipynb)      
使用tf.function可以将普通的python函数转换为tensorflow的图结构，从而提高代码运行效率    

* 显示转换  
    ```python
    # 先定义一个普通的python函数 
    def scaled_elu(z, scale=1.0, alpha=1.0):
        # z >= 0 ? scale * z : scale * alpha * tf.nn.elu(z)
        is_positive = tf.greater_equal(z, 0.0)
        # 在tf中，三元表达式用where实现
        return scale * tf.where(is_positive, z, scale * alpha * tf.nn.elu(z))

    print(scaled_elu(tf.constant(-3.)))
    print(scaled_elu(tf.constant([-3., -2.5])))

    # 显示转换为tf.function 
    scaled_elu_tf = tf.function(scaled_elu)
    print(scaled_elu_tf(tf.constant(-3.)))
    print(scaled_elu_tf(tf.constant([-3., -2.5])))

    print(scaled_elu_tf.python_function is scaled_elu)
    ```  

* 使用annotation转换   
    ```python
    # 计算1 + 1/2 + 1/2^2 + ... + 1/2^n

    @tf.function
    def converge_to_2(n_iters):
        total = tf.constant(0.)
        increment = tf.constant(1.)
        for _ in range(n_iters):
            total += increment
            increment /= 2.0
        return total

    print(converge_to_2(20))
    ```   

* 从ConcreteFunction获取图结构信息    
    使用tf.function转换为图结构后，可以在function注解中使用`input_signature`限定输入参数类型，同时也可以将当前函数转换为可以保存为savedModel的图结构，在模型复原时，可以通过该savedModel还原出图中的某个op或者tensor    

    ```python   
    @tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])
    def cube(z):
        return tf.pow(z, 3)

    try:
        # 如果输入参数类型与input_signature不符，会抛出异常
        print(cube(tf.constant(1., 2., 3.)))
    except ValueError as ex:
        print(ex)
    print(cube(tf.constant([1, 2, 3])))

    # @tf.function py func -> tf graph
    # get_concrete_function -> add input_signature -> savedModel
    # 获取ConcreteFunction对象
    cube_func_int32 = cube.get_concrete_function(
        tf.TensorSpec([None], tf.int32))    

    # 打印当前的图结构信息 
    cube_func_int32.graph.as_graph_def()
    # 通过name获取op或者tensor，这个name可以在定义时指定
    cube_func_int32.graph.get_operation_by_name('x')
    cube_func_int32.graph.get_tensor_by_name('x:0')
    ```  

### [tf.GradientTape实现自定义求导](./tf_diffs.ipynb)    
如果多次使用tape求导，需要在定义时将`persistent`参数设置为True   
* tf.GradientTape求一阶导    
    ```python
    x1 = tf.Variable(2.)
    x2 = tf.Variable(3.)
    with tf.GradientTape(persistent=True) as tape:
        z = g(x1, x2)
    dz_x1 = tape.gradient(z, x1)
    dz_x2 = tape.gradient(z, x2)
    print(dz_x1, dz_x2)

    # 手动释放tape
    del tape
    ```    

* tf.GradientTape嵌套求二阶导数   
    ```python
    x1 = tf.Variable(2.)
    x2 = tf.Variable(3.)
    with tf.GradientTape(persistent=True) as outer_tape:
        with tf.GradientTape(persistent=True) as inner_tape:
            z = g(x1, x2)
        inner_grads = inner_tape.gradient(z, [x1, x2])
    outer_grads = [outer_tape.gradient(inner_grads, [x1, x2])
                for inner_grad in inner_grads]
    print(outer_grads)
    del inner_tape
    del outer_tape
    ```   

* tf.GradientTape结合keras.optimizers使用   
    ```python
    learning_rate = 0.1
    x = tf.Variable(0.0)

    opt = keras.optimizers.SGD(learning_rate)

    for _ in range(100):
        with tf.GradientTape() as tape:
            z = f(x)
        dz_dx = tape.gradient(z, x)
        # optimizers.apply_gradients方法传参是一个列表，列表元素是tuple，每个tuple对应一个目标变量和它的导数
        opt.apply_gradients([(dz_dx, x)])
    print(x)
    ```


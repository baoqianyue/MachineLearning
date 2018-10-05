## Tensorflow 基础函数    

### tf.argmax()   

类似于`np.argmax()`,返回最大数值所在的下标      

### tf.equal()    

对比两个矩阵或向量相等的元素，如果相等对应位置上为True，反之为False，返回的矩阵维度与比较的矩阵一致      

```python
A = [[1,3,4,5,6]]
B = [[1,3,4,3,2]]

with tf.Session() as sess:
    print(sess.run(tf.equal(A, B)))
```    

输出：  
```
[[ True  True  True False False]]
```   

### tf.cast()     

`tf.cast(x, dtype)`将x的数据格式转换为dtype       

```python
a = tf.Variable([1,0,1,0,1])
b = tf.cast(a, dtype=tf.bool)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(b)) 
```  
输出：    
```
[True, False, False, True, True]
```     

### tf.truncated_normal()   

`tf.truncated_normal(shape, mean, stddev)`生成截断式的正太分布数据。         
shape表示生成tensor的维度，mean是均值，stddev是标准差      
正态分布有$3/sigma$分布，如果随机生成的数据与均值的差值大于两倍的标准差，就重新生成。     
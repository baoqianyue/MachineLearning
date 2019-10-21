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
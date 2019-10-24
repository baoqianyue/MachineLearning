## tf.data基础API    

### [tf.data.Dataset.from_tensor_slices](./tf_data_basic_api.ipynb)     

该函数可以接受numpy、python元组、python字典等类型的数据   
* 构建数据集   
    ```python
    # 将numpy类型数组转为datasets
    dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
    print(dataset)
    # 遍历输出item，每个item都是tensor类型
    for item in dataset:
        print(item) 
    ```    

* repeat和batch操作   
    ```python
    # 对当前数据集重复三个epoch，生成batch的batch_size为5  
    # 这些操作可以链式调用，每次调用结束后都会生成一个新的Dataset
    dataset = dataset.repeat(3).batch(5)
    for item in dataset:
        print(item)
    ```   

* interleave操作  
    在dataset内部做map转换，常见的操作是：文件名dataset -> 带有图片的具体数据集
    
    ```python
    # 该函数三个参数 
    # 参数map_fn：表示对数据集的转换操作
    # 参数cycle_length:数据变化并行程度
    dataset2 = dataset.interleave(
        lambda x : tf.data.Dataset.from_tensor_slices(x), # 对每个item生成一个新的数据集
        cycle_length=5, 
        block_length=5
    )
    for item in dataset2:
        print(item)
    ```

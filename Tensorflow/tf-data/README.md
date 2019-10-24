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

### [手动生成csv文件](./tf_keras_generate_csv.ipynb)    
将房价预测数据集分parts写入到csv文件中   
```python
output_dir = 'generate_csv'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def save_to_csv(output_dir, data, name_prefix, header=None, n_parts=10):
    """
    data:全部数据，包括features和label
    name_prefix:用来区分train、test、valid
    header:csv文件中header
    n_parts:表示该数据集分成几个csv文件存储
    """
    # 先确定每个csv文件路径
    path_format = os.path.join(output_dir, "{}_{:02d}.csv")
    filenames = []
    # 循环中首先将所有数据行分成n_parts组，然后对每个组取出他们在原data中的indexes
    for file_idx, row_indexes in enumerate(
        np.array_split(np.arange(len(data)), n_parts)):
        part_csv_path = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv_path)
        with open(part_csv_path, 'wt', encoding='utf-8') as f:
            if header is not None:
                f.write(header + '\n')
            for row_index in row_indexes:
                f.write(','.join(
                    [repr(col) for col in data[row_index]]))
                f.write('\n')
    return filenames

# 组装数据，将features和label拼接到一块
train_data = np.c_[X_train_scaled, y_train]
valid_data = np.c_[X_valid_scaled, y_valid]
test_data = np.c_[X_test_scaled, y_test]

# 构建csv的header
header_cols = housing.feature_names + ['MidianHouseValue']
# 将list连接成字符串
header_str = ','.join(header_cols)

train_filenames = save_to_csv(output_dir, train_data, 'train',
                              header_str, n_parts=20)
test_filenames = save_to_csv(output_dir, test_data, 'test',
                             header_str, n_parts=10)
valid_filenames = save_to_csv(output_dir, valid_data, 'valid',
                              header_str, n_parts=10)
```

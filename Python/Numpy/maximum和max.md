## np.maximum     

`np.maximum(X,y)`       
是将X和y逐位比较取大者                   

```Python
import numpy as np
print(np.maximum([[0,1,2],[3,5,6]], 3)) # 表示与[[3,3,3],[3,3,3]]进行逐位比较
print()
print(np.maximum([[0,1,2],[3,5,6]], [[1,3,0],[4,1,8]]))
```

输出：
```
[[3 3 3]
 [3 5 6]]

 [[1 3 2]
 [4 5 8]]
```     

* 使用np.maximum实现Relu函数     

    `relu`函数表达式：    

    $$f(x) = max(0,x)$$        

    ```python
    def relu(x):
        return np.maximum(0, x)
    ```     


## np.max      

`def amax(a, axis=None, out=None, keepdims=np._NoValue)`       
求序列的最值，`axis`默认为列方向，`axis=1`时为行方向的最值           
`keepdims`的取值控制输出的是一个numpy矩阵还是一个向量           

```Python
import numpy as np
print(np.max([[0,1,2],[3,5,6]], axis=0, keepdims=False))
print()
print(np.max([[0,1,2],[3,5,6]], axis=1, keepdims=False))
print()
print(np.max([[0,1,2],[3,5,6]], axis=1, keepdims=True))
```

输出：   

```
[3 5 6]

[2 6]

[[2]
 [6]]
```


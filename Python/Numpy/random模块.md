## np.random.chioce      

可以从一个int数字或一个1维array中随机选取内容，并将结果放置在n维的数组中返回      

```
def choice(a, size=None, replace=True, p=None)
```    
`replace`参数确定是否是放回的选取，如果取值为`True`表示有放回，随机选取的结果中有重复的，反之为无放回的选取      

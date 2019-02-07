# 统计学习方法概论   

## 问题的类型    

* 输入变量和输出变量都为连续变量的预测问题称为回归问题   

* 输出变量为有限个离散变量的预测问题称为分类问题   

* 输入变量和输出变量都为变量序列的预测问题为标注问题      

## 联合概率分布   

统计学习假设数据存在一定的统计规律，监督学习假设输入和输出的随机变量$X$和$Y$遵循联合概率分布$P(X, Y)$,$P(X, Y)$表示分布函数或者分布密度函数，**在模型的学习过程中，假定这一联合概率分布存在**   


监督学习的模型可以是概率模型或者非概率模型，由条件概率分布$P(Y|X)$或者决策函数(decision function)$Y=f(X)$表示，随具体的学习方法而定，对于具体的输入进行相应的输出预测时，写作$P(y|x)或者y=f(x)$，x和y小写表示具体的值     

## 统计学习三要素   

`方法 = 模型 + 策略 + 算法`      

### 模型    

首先要考虑的是学习什么样的模型，这里的模型可以直接看成上面提到的条件概率分布或者决策函数   

由**决策函数表示的模型为非概率模型，由条件概率表示的模型称为概率模型**     

### 策略     

分为经验风险最小化和结构风险最小化    

* 经验风险最小化(empirical risk minimization)    

    经验风险是**相对于一个训练数据集**的平均损失     

    所以按照经验风险最小化策略求最优模型就是求解最优化问题：   

    $$\frac{1}{N}\sum_{i=1}^{N}L(y_i, f(x_i))$$    

    当样本容量较大时，经验风险最小化策略可以保证很好的学习效果，比如，极大似然估计(maximum likelihood estimation)就是经验风险最小化的一个例子     

    但是，当样本的容量很小时，容易出现过拟合现象(over-fitting)    
* 结构风险最小化(structural risk minimization)    

    该策略是针对于过拟合现象提出的，在原有经验风险最小化的基础上加上了针对具体模型的正则化项或者惩罚项      

    结构风险的定义：   

    $$\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i)) + \lambda{J(f)}$$     

    其中$J(f)$表示模型的复杂度，复杂度表示了对复杂模型的惩罚，$\lambda >= 0$是系数，用来权衡经验风险和模型复杂度      


## 模型评估和模型选择    

### 训练误差和测试误差       

* 训练误差的大小反映出给定的问题是否是一个容易学习的问题    

* 测试误差反映了当前学习方法对于未知的测试数据集的预测能力，这个测试能力通常称为泛化能力(generalization ability)     

### 过拟合和模型选择     

为了一味追求对训练数据的预测能力，无疑会提高模型的复杂程度，随之带来的是模型的参数增多，这将导致模型对于已知数据预测的很好，但是对于未知数据的预测很差，这种现象就是过拟合(over-fitting)    

**可以说模型选择旨在避免过拟合并提高模型的预测能力**    

下面使用一个多项式函数拟合实例来说明过拟合和模型选择      

假设给定一个训练数据集:   

$$T = \{(x_1, y_1),(x_2, y_2),...,(x_N, y_N)\}$$   

假定给定10个数据点，这10个数据点符合$sin(2\pi x)$分布，然后给每一个观测点加上了一个高斯分布的随机噪声，再使用$M$次多项式对这10个数据点进行拟合    

设$M$次多项式为：   

$$f_M(x, \omega) = \omega_0 + \omega_1x + \omega_2x^2 + ... + \omega_Mx^M = \sum_{j=0}^{M}\omega_jx^j$$      

该式中$x$为单变量输入，$\omega_M$是M+1个参数     

首先先确定模型的复杂度，即确定多项式的次数，然后根据得到的模型复杂度，按照经验风险最小化策略，求解参数，就是多项式各项的系数(包括常数项)   


使用最小二乘法求得令残差平方和最小的模型参数     

> 高斯于1823年证明了最小二乘法的一个最优性质：在所有无偏的线性估计类中，最小二乘法方法具有最小的方差    

最优化下式：   

$$min\sum_{i=1}^{N}(\sum_{j=0}^{M}\omega_jx_i^j - y_i)^2$$  

```python  
def goal_fun(x):
    """
    目标函数
    :param x:
    :return: 目标函数设置为y = sin2Πx
    """
    return np.sin(2 * np.pi * x)


def fit_fun(p, x):
    """
    多项式函数
    eg: np.poly1d([1, 2, 3])生成1x^2 + 2x^1 + 3x^0
    :param p: 各次项前面的系数或者说是权重
    :param x:
    :return:
    """
    f = np.poly1d(p)
    return f(x)


def residual_fun(p, x, y):
    """
    计算残差
    :param p:
    :param x:
    :param y: y为观测值(真实值)
    :return:
    """
    res = fit_fun(p, x) - y
    return res


def fitting(x, y, x_points, M=0):
    """
    使用最小二乘法进行拟合并进行可视化
    :param x: 观测点的x坐标
    :param y: y坐标
    :param x_points: 可视化的x坐标
    :param M: 最高次数
    :return:
    """
    # 随机初始化多项式各项系数, M+1表示有一个常数项
    p = np.random.rand(M + 1)
    # 最小二乘法， 传入残差计算函数
    p_lsq = leastsq(residual_fun, p, args=(x, y))
    print('fitting param:', p_lsq[0])

    # 可视化
    plt.title('M={}'.format(M))
    plt.plot(x, y, 'ro', label='noise')
    plt.plot(x_points, goal_fun(x_points), label='goal')
    plt.plot(x_points, fit_fun(p_lsq[0], x_points), label='fitted curve')
    plt.legend()
    plt.show()
    return p_lsq


if __name__ == '__main__':
    # 给定10个数据点
    x = np.linspace(0, 1, 10)
    y_ = goal_fun(x)
    # 给10个坐标点加入随机噪声
    y = [i + np.random.normal(0, 0.1) for i in y_]
    # 可视化点x坐标
    x_points = np.linspace(0, 1, 1000)
    # M = 0
    p_lsq_0 = fitting(x, y, x_points, M=0)
    # M = 1
    p_lsq_1 = fitting(x, y, x_points, M=1)
    # M = 3
    p_lsq_3 = fitting(x, y, x_points, M=3)
    # M = 9
    p_lsq_9 = fitting(x, y, x_points, M=9)
```    

M=1的拟合结果：    
![M=0](img/m1.png)     

M=3的拟合结果：   
![M=3](img/m3.png)  

M=9的拟合结果：  
![M=9](img/m9.png)    

* 分析     

    * M=1时拟合效果并不好，原因是模型较为简单，参数较少，无法描述训练数据集     

    * M=9时，多项式曲线可以通过每一个数据点，训练误差为0，数据拟合的效果是最优的，但是因为训练数据有随机噪声，这种模型对于未知数据的预测能力很差，容易发生过拟合现象     

    * M=3时，多项式曲线对训练数据的拟合效果足够好，而且模型也比较简单，实际的效果较好      

    * **模型在选择时，不仅要考虑模型对已知数据的预测能力，还有考虑对未知数据的预测能力**    

    * 随着多项式次数(模型复杂度)的增加，训练误差会逐渐减小，一直趋近于0，但是测试误差会先减小然后再增大，**但是我们的目标是让训练误差较小时，测试误差也较小，这就需要用到模型的选择方法**  


### 正则化和交叉验证     

上例中当提高超参数M的值，会发生越来越严重的过拟合现象，也就是测试误差会先减小后增大，所以我们是要使用模型选择方法来寻找到使得测试误差较小的模型  

这里先使用正则化     

* 正则化(regularization)   

    正则化是结构化风险最小化策略的实现，就是在经验风险上加上一个惩罚项(penalty term)     

    正则化项通常是模型复杂度的单调递增函数，模型越复杂，正则化值越大，比如正则化项可以是模型参数向量的范数      

    正则化的一般形式为：   

    $$min \frac{1}{N} \sum_{i=1}^{N}L(y_i,f(x_i)) + \lambda{J(f)}$$    

    其中第一项是经验风险(训练误差),第二项是正则化项，`λ>=0`为调整两者之间关系的系数    

    正则化项可以取不同的形式，比如回归问题中，损失函数是平方损失，正则化项就可以是参数向量`ω`的L2范数   

    $$L(\omega) = \frac{1}{N} \sum_{i=1}^{N}(f(x_i;\omega) - y_i)^2 + \frac{\lambda}{2}||\omega||^2$$

    这里`||ω||`是参数向量的L2范数    

    **上式的形式代表了当前的损失函数，第一项代表了当前的经验风险，经验风险较小的模型可以较复杂(拥有很多的非零参数),这时，第二项的模型复杂度(参数向量的范数)会较大，正则化的作用就是选择经验风险与模型复杂度同时较小的模型**   

    正则化符合奥卡姆剃刀(Occam's razor)原理      

    > entia non sunt multiplicanda praeter nesessitatem (如无必要勿增实体)   

    正则化项对应于模型的先验概率    

    下面给上面的实例应用正则化，降低模型的过拟合程度     

    ```python
    def residual_with_regularization(p, x, y, regularization=0.0001):
        """
        在误差计算时加上正则项
        :param p:
        :param x:
        :param y:
        :return:
        """
        # 计算基本误差
        res = fit_fun(p, x) - y
        # 加上正则项
        res = np.append(res, np.sqrt(0.5 * regularization * np.square(p)))
        return res


    def fitting_with_regularization(x, y, x_points, M):
        # 随机初始化各项的系数，M + 1代表初始化偏置项常数
        p_init = np.random.rand(M + 1)
        p_lsq_with_regularization = leastsq(residual_with_regularization, p_init, args=(x, y))
        plt.plot(x_points, goal_fun(x_points), label='goal')
        plt.plot(x_points, fit_fun(p_lsq_9[0], x_points), label='fitted curve')
        plt.plot(x_points, fit_fun(p_lsq_with_regularization[0], x_points), label='regularization')
        plt.plot(x, y, 'ro', label='noise')
        plt.legend()
        plt.show()
        return p_lsq_with_regularization
    ```  

    加上正则项后的拟合效果：  
    ![withreg](img/withreg.png)   


* 交叉验证(Cross Validation)      

    当给定的样本数据较为充足时，可以将全部样本按照一定比例切分成三部分，分别为训练集(training set),验证集(validation set),测试集(test set)    

    训练集用来训练模型，验证集用来进行模型的选择，测试集用来评估模型   

    但是很多实际情况中的样本容量很小，直接对全部样本进行切分并不能得到较好的效果，所以要对切分后的各部分再进行组合   

    * K折交叉验证(K-fold cross validation)     

        首先将所给全部数据其分成K个互不相交的大小相同的子集，然后使用K-1个子集的数据训练模型，利用余下的子集测试模型    

        然后将这个过程对所有可能的K种选择重复进行，最后选出K次评测中平均测试误差最小的一个模型   


### 分类问题的评估标准   

分类是监督学习中的一个核心问题，在监督学习中，数据变量Y取值为有限个离散值时，预测问题就变成了分类问题   

评价分类模型的指标一般是分类准确率(accuracy)，给定测试数据集，分类器正确分类的样本数和总样本数的比值    

这里说明一下二类分类问题的常用评价指标     
分类器在同一个测试集上进行预测可能出现以下四种情况：   

* TP 将正类分类成正类      
* FN 将正类分类成负类      
* FP 将负类分类成正类     
* TN 将负类分类成负类      

二分类问题中常用的评价指标是精确率和召回率      

* 精确率(precision)    

    $$P = \frac{TP}{TP + FP}$$     

* 召回率(recall)    

    $$R = \frac{TP}{TP + FN}$$     

* F1值    
    是精确率和召回率的调和均值   

    $$
    \frac{2}{F_1} = \frac{1}{P} + \frac{1}{R}    \\    
    F_1 = \frac{2TP}{2TP + FP + FN}
    $$  

当精确率和召回率都高时，F1值也会高   
 

## CycleGAN   

### 概述   

该网络使用无监督训练(Unsupervised training)方式。  

数学形式表达：   
X和Y是两种不同类型的图像分布集，我们需要寻找一个映射$G:X -> Y$使得$y^{hat}=G(x)$   

该网络使用的数据集不是成对（paird data）数据集，所以无监督训练学习到的$y^{hat}$分布中含有无数种可能的映射，因为不能保证这种混合型的数据集中任意一个x对任意一个y的映射都有意义，所以训练过程中也会出现模式崩塌(mode collapse)的问题，为了解决该问题，CycleGAN中加入了图像转换中的一个特性来作为判别标准，**循环一致性(cycle consistent)**，比如将一个英文句子使用生成器G翻译成汉语，然后将汉语句子使用生成器F翻译回英语，理想状况下这两个英文句子应该是一样的，我们最优化的目标就是让这两个句子的差距最小。    

数学表示：  

英语到汉语翻译器G,x为英语句子    

$$
x -> G(x) -> F(G(x)) = x  
$$  

汉语到英语翻译器F，y为汉语句子   

$$
y -> F(y) -> G(F(y)) = y  
$$    

### 网络结构和损失函数   

* 网络结构概览   

    * 生成器使用残差网络，判别器使用PatchGAN    

    * 一共有两个生成器(G, F)和两个判别器($D_Y$, $D_X$)   
        * G将X domain的图像转换成Y domain    
        * F将Y domain的图像转成成X domain    
        * $D_X$判别X domain中的图像真假  
        * $D_Y$判别Y domain中的图像真假    
    
* 生成器结构    
    生成器包含三个成分   
    * Encoder  
        Encoder相当是特征提取部分，假设输入图像为(256，256，3)   
        具体op为三个卷积层，Encoder输出向量维度为(64，64，256)    
       
    * Transformer    
        Transformer就是特征映射模块，输入向量为经过卷积层提取到的维度为（256，256，3）的特征，为了实现CycleGAN从DomainA到DomainB的转换，这里就需要对特征向量进行映射，为了保证映射后的特征丢失太多的输入图像信息，这里使用6层[Res-Block](../Cnn/卷积神经网络进阶(Alex,Vgg,ResNet).md)来做特征映射，输出向量的维度仍为(256,256,3)    

    * Decoder     
        Decoder的相当于是Encoder的逆向过程，使用反卷积，最后输出的图像就是生成图像，维度为(256,256,3)   

* 判别器结构   

    这里的判别器和普通GAN的判别器相同，使用5个卷积层将生成器的生成的图像进行下采样，然后输出一维向量     

* 结构总结   
    
    根据上面定义的生成器和判别器，为了实现CycleGAN的结构，我们需要两个映射方向的GAN结构：   
    * A -> B：Generator(A->B), Discriminator(A)    
    * B -> A: Generator(B->A), Discriminator(B)   


* 损失函数    

    * Discriminator loss    
        * Part1   
            当给判别器输入DomainA中的真实图像时，判别器输出应该为接近于1，loss应该尽可能小，形式化表达为最小化$(Discriminator_A(a) - 1)^2$，对于loss_B是同样的，下面为loss示例代码：  
            ```python
            # dec_A为判别器A的输出
            D_A_loss_1 = tf.reduce_mean(tf.squared_difference(dec_A, 1))
            D_B_loss_1 = tf.reduce_mean(tf.squared_difference(dec_B, 1))
            ```    

        * Part2   
            当然，判别器应该能够分辨出生成器产生的图像，当输入到判别器中的图像为生成器生成图像时，判别器输出应该接近于0，形式化表达为最小化$(Discriminator_A(Generator_{B->A}(b)))^2$，示例代码如下：  
            ```python
            # dec_gen_A表示生成器Generator(A->B)的生成图像输入到dec_A判别器的输出结果    
            D_A_loss_2 = tf.reduce_mean(tf.square(dec_gen_A))
            D_B_loss_2 = tf.reduce_mean(tf.square(dec_gen_B))   

            D_A_loss = (D_A_loss_1 + D_A_loss_2) / 2
            D_B_loss = (D_B_loss_1 + D_B_loss_2) / 2
            ```  

    * Generator loss     
        理想生成器应该能够迷惑判别器，即生成器生成的图像输入到判别器后，判别结构接近于1，形式化表达为最小化$(Discriminator_B(Generator_{A->B}(a)) - 1)^2$，代码示例如下：   
        ```python
        g_loss_B_1 = tf.reduce_mean(tf.squared_diffierence(dec_gen_B, 1))
        g_loss_A_1 = tf.reduce_mean(tf.squared_diffierence(dec_gen_A, 1))
        ```     

    *  Cyclic loss    
        循环loss，也是该网络的核心，我们使用生成器生成的图像输入到相反映射的生成器中，输出图像应和原始图像相近   

        ```python
        # cyc_A为生成器B的输出图像输入到B->A生成器后产生的图像
        cyc_loss = tf.reduce_mean(tf.abs(input_A - cyc_A) + tf.reduce_mean(tf.abs(input_B, cyc_B)))      

        g_loss_A = g_loss_A_1 + 10 * cyc_loss 
        g_loss_B = g)loss_B_1 + 10 * cyc_loss   
        ```   
        参数10表示cycloss比重更大   


### 训练细节    


### CycleGAN 的缺点    

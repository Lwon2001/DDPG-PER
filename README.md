# DDPG-PER
DDPG with Prioritized Experience Replay
paper "Continuous control with deep reinforcement learning"

在手动实现了sumtree结构的基础上参考paper实现了PER，将其用在DDPG中，并在同等参数下测试PER的效果。

实验证明，PER对于加速网络收敛确实有一定的作用

不采用PER的DDPG在150 episode处接近收敛：

![image](https://user-images.githubusercontent.com/59995175/193972512-b31580f3-1c7a-492c-8e53-885971541d0a.png)

采用PER的DDPG在80 episode处接近收敛，其中PER的超参部分采用原paper中所用参数：
![6T2FRHG`4{$EB{6S)D4)1SJ](https://user-images.githubusercontent.com/59995175/193972601-180ccac6-2597-433e-b310-9d7f7e755884.png)

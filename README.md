安装包：
python3
pytorch
gym

注意事项:
1.
连续动作的游戏在windows环境下安不上，gym对windows的支持不完全
有linux的同学可以试一下，只需要pip install gym[游戏名] 或 pip install gym[all]
2.
所有的trick都实现了, 大家可以用true/false控制, 我自己测试的没报错
大家也可以测试一下, 但是用了之后效果好不好我就不知道了(其实写的对不对我也不知道)
3. 
train.py就是训练的main; test.py就是测试的main; ppo.py就是AI; 结构很简单
4.
目前只有log和model，后期有时间会把plot和gif也补上


<h1 align="center">
电影推荐系统
</h1>
<p align="center">
    Project of AI3602 Data Mining, 2025 Spring, SJTU
    <br />
    <a href="https://github.com/zzctmd"><strong>Zichen Zou</strong></a>
    &nbsp;
    <a href="https://github.com/leothehuman327"><strong>Yiming Chen</strong></a>
    <br />
</p>

## Abstract
<img src="assets/sampled/picture.png" height="500px"/>  
🎬 欢迎使用我们的电影推荐系统! 

本项目是一个基于LightFM的电影推荐系统。系统利用MovieLens数据集中的用户评分数据和电影特征信息,构建了一个混合推荐模型。该系统可以根据用户喜欢的电影的类型、年代等特征,为用户推荐个性化的电影内容。系统的主要特点包括:

- 🚀 支持冷启动推荐,即使对新用户也能基于电影特征进行推荐
- 🎯 考虑了电影的多个维度特征,包括类型、上映年份等
- ⚡️ 采用WARP损失函数优化模型,提高推荐准确性
- 🖥️ 提供了友好的Web界面,方便用户交互使用

通过该系统,用户可以快速发现符合自己兴趣的优质电影内容,获得个性化的观影体验。让我们一起开启奇妙的电影之旅吧! 🎉


## 🛠️ Requirements
To run this project, please run the following commands:
```
conda env create -f environment.yml
```

- Download parameters of the model from [交大云盘](https://pan.sjtu.edu.cn/web/share/f099dbf67a3b3c62849ebb315ea2e35a) and put it into the `model` folder
- Download the MovieLens Latest Dataset (ml-32m folder) from [交大云盘](https://pan.sjtu.edu.cn/web/share/f099dbf67a3b3c62849ebb315ea2e35a) and put it into the `AI3602` folder

## 🚀 Training
1. Run the following commands to start training:

```bash
python train.py  --movie_csv  ./ml-32m/movies.csv   --ratings_csv  ./ml-32m/ratings.csv 
```
Check the results in the folder.


## 💡 Inference
Here are the instructions: 
```bash
python app.py
```
You can experience our demo showcase and achieve the same effect as shown in the video below.




<video src="https://github.com/user-attachments/assets/89111b27-9dae-4dba-a2bf-2502396dfa7d" controls="controls" width="74" height="48"></video>


## Contact
If you have any questions, please contact us via 
- zzcnb123456@sjtu.edu.cn









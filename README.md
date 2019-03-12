# tf-yolov3

- #### 1.首先你需要有一个linux环境以及1080以上的GPU
```
ssh root@192.168.**.**
```
- #### 2.用Miniconda建一个虚拟环境
[下载Miniconda](https://conda.io/en/latest/miniconda.html "下载Miniconda")
点击`linux` --> `python3.7`--> `64bit`
- #### 3. conda一个新环境
```
conda create -n tf-yolo python=3.6
```
  tf-yolo 是环境的名字，中间，会输入一下 y，才能继续安装。
  <br/>
   conda 常用命令：
  ` conda env list` &#8195;&#8195; 列举当下的conda虚拟环境
  `conda remove -n tf-yolo --all` &#8195;&#8195;删除环境
- #### 4. 激活环境
```
source ~/.bashrc    
source activate tf-yolo  
# source deactivate     #退出当前环境
```
- #### 5. 下载源码
```
git clone https://github.com/juciny/tf-yolov3.git
# 从清华源下载安装包，如果出现安装失败，换个版本就好
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```
- #### 6. 获取训练模型
下载 [yolov3.weights](https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3.weights "yolov3.weights")，该权重是用coco数据集训练的，可以识别80种常见的物体
将.weights文件  转换为.ckpt文件和.pb文件
```
python convert_weight.py --convert --freeze
```

 转换模型格式说明

|模型格式|说明|
|:-----  |-----                           |
| .ckpt.meta  |计算图的结构，没有变量的值（可以在tensorboard / graph中看到）  |
|  .ckpt.data |包含所有变量的值，没有结构  |
|  .ckpt.index |映射前两个文件的索引（通常不是必需的） |
|  .pb |可以保存整个图表（元+数据） |
- #### 7. 开始测试
```
python run_photo.py
```


####训练过程可以分为3个step：
- 数据处理部分：
    1. 爬数据或下载数据集  
	2. 标注数据   
	3. 转换数据为tfrecord格式
- 训练部分：
	 4. 生成先验框
	 5. 获取预训练权重
	 6. 开始训练
- 测试部分：
	 7. 将CKPT转为PB文件
	 8. 测试

这次我们的训练目标就是小猪佩奇

- #### 1.爬数据或下载数据集
   从[小猪佩奇贴吧](https://tieba.baidu.com/p/5287936288 "小猪佩奇贴吧")，爬图片作为数据集，[爬虫代码](https://blog.csdn.net/eereere/article/details/88389384 "爬虫代码")。

- #### 2.标注数据
由于[标注工具](https://github.com/tzutalin/labelImg "标注工具")需要可视化的界面，所以需要在本地运行，直接将代码下载到本地。
在运行代码之前，需要安装两个依赖包：
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyqt5
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lxml
cd labelImg-master
python labelImg.py
```

labelImg的快捷键：

    `w`——创建一个bounding box
    `s`——上一张图片
    `d`——下一张图片
    `ctrl`+`s`——保存
注意：我在这里 只标注了 小猪佩奇的头，另外，在标注过程中，bbox的要尽量紧挨着目标物体，标注结束后，会在保存的文件夹中，生成一堆xml文件。

- #### 3.转换数据为tfrecord格式
 网络训练，最终需要转换的格式为tfrecord，而我们的标注文件为xml
 数据转换的过程为：xml  ---> txt --->tfrecord
 在YOLOv3的根目录下新建一个文件夹pig_dataset，用于存放小猪佩奇的数据集
 --tf-yolov3
 -----pig_dataset
 ---------images     # 用于存放小猪佩奇的图片
 ---------labels.txt   # 用于存放 图像的labels
```
python xml_to_txt.py  # 将xml转换为txt文件
sh ./scripts/make_tfrecords.sh     # 将txt文件转为tfrecord文件
```

- #### 4.生成先验框
```
python kmeans.py --dataset_txt ./pig_dataset/train.txt --anchors_txt ./data/pig_anchors.txt
```
参数说明：
&#8195;&#8195;--dataset_txt    训练集的txt路径
&#8195;&#8195;--anchors_txt    先验框的保存路径


- #### 5.获取预训练权重
我们之前下载的是yolov3.weight，这个只是网络的权重，而没有模型的网络训练要加载.ckpt格式的预训练权重，因此，需要将yolov3.weight转为.ckpt。
```
python convert_weight.py --convert
```
转换之后会生成三个文件，.ckpt.meta 、.ckpt.data 、.ckpt.index。

- #### 6.开始训练
在运行训练脚本之前，要修改train.py中的几个参数：
  &#8195;&#8195;17行 CLASSES——类别名称 的路径
  &#8195;&#8195;18行 ANCHORS——先验框的路径
  &#8195;&#8195;22行 train_tfrecord——训练集的路径
  &#8195;&#8195;23行 test_tfrecord——测试集的路径
  &#8195;&#8195;73行 save_path   ——模型保存的路径
 ```
 python train.py
 ```

- #### 7.将ckpt转换为pb
```
python convert_weight.py -cf ./checkpoint/pig-yolov3.ckpt-1500 -nc 1 -ap ./data/pig_anchors.txt --freeze --pb_file ./checkpoint/yolov3_pig
```
参数说明：
 &#8195;&#8195;-cf   &#8195;&#8195;  ckpt file的路径
&#8195;&#8195;-nc  &#8195;&#8195; 类别的数目
&#8195;&#8195;-ap  &#8195;&#8195; 先验框的txt路径
&#8195;&#8195;--freeze  &#8195;&#8195; Bool型，表示freeze模型
&#8195;&#8195;--pb_file &#8195;&#8195;  生成pb模型的路径（包括前缀名）
生成结果：
在./checkpoint文件夹下，会生成两个pb文件：yolov3_pig_cpu.pb  yolov3_pig_gpy.pb

- #### 8. 测试
`python run_photo.py`

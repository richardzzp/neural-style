## 项目结构

contents： 存放原始图片，命名风格为[数字]-content.jpg

styles: 存放风格图片，命名风格为[数字]-style.jpg

outputs： 存放风格迁移后的输出文件，命名风格为[数字]-output.jpg。

以上三个文件夹中的文件通过[数字]来关联。

neural_style.py:  定义了函数的主要参数，包含对图像的读取和存储，读取完成后传入stylize.py中进行计算。

stylize.py： 核心代码，包含了训练、优化等过程。

vgg.py： 定义了网络模型及相关的运算





## 依赖安装

在本项目根目录下运行 

```
pip install -r requirements.txt
```

即可自动安装依赖。另外需要自行下载imagenet-vgg-verydeep-19.mat放在根目录

## 运行

在根目录下输入

```
python neural_style.py
```

即可运行，如果GPU加速不支持（会导致运行速度大幅度减慢），建议使用谷歌Colaboratory。

### 
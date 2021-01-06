# quality_evaluate
行人质量评价模块**（使用固化模型推理**）是根据行人图片的模糊分数、宽高比、分辨率、截断与朝向方面，综合评价行人图片的质量，并给出质量分数。
##### 功能性需求
1、提供图片质量评分功能，对每张图片进行评分

2、提供尺寸过滤功能，过滤不满足要求的图片

3、使用矩阵运算实现各维度评估工作，减少图片评估处理耗时

##### 例子

在本目录下提供了一个简单的例子`test.py`，测试数据存放于`./../test_data`目录下，安装`requirements.txt`中的包后，运行指令`python3 test.py`

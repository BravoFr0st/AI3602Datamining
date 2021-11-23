## For T.A.

Go and Run  `python ./src/main.py` **under this file** (`519030910346_EXP1`).

Will get `./res.csv`.

# Data Mining HW1： CommunityDetection

## 任务目标

在提供的图数据上实现社区检测算法。在此次实验中，我们会提供一个包含31136个节点、16万条边左右的有向引文网络，其中有向边的方向代表论文的引用关系。这些论文根据其会议名称被分为5类（AAAI, IJCAI, CVPR, ICCV, ICML），分别用0-4来表示。你的任务是应用社区检测算法将这些论文分为五类。

## 文件组织形式

- `./data` : 数据集，包括`edges.csv`,`ground_truth.csv`,`sample_res.csv`
  - `edges_update.csv`:每行代表源节点和目标节点（source and target node）。
  - `ground_truth.csv`: 测试数据，同时方便大家给每个类一个标签（0-4），约占总节点数1%，包含每类会议论文中的60个节点。
  - `sample_res.csv`: 参考输出文件。
- `./src`: 代码文件。在根目录下运行`python ./main.py`会生成`./res.csv`.
  - `main.py`: 主代码.
  - `utils.py`：两个工具函数.
- `res.csv`: 预测结果. 要求其第一列为节点id，第二列为算法预测出的节点类别category.
- `./requirements.txt`
- `./README.md`

## 评分方式

我们使用如下方程来计算模型的准确率，其中$N$代表节点总数，$N_{true}$代表节点预测正确的数量。
$$ Accuracy = \frac{N_{true}}{N}$$

## 提交：

你需要提交一个压缩文件包，提交至canvas平台的作业实验一中，命名格式为：学号_EXP1.zip，其中仅包含一个同名文件夹。为了方便我们复现代码，里面的文件夹需为如下格式：“学号_EXP1/src” （包含代码的文件夹），“学号_EXP1/data”（包含数据集的文件夹），“学号_EXP1/readme.txt” 和 “学号_EXP1/requirements.txt”。**我们会运行 “学号_EXP1/src/main.py”来测试代码，main.py需要在其路径下生成预测结果保存在“res.csv“中，其第一列为节点id，第二列为算法预测出的节点类别category。** 请注意代码中的路径为相对路径。提交截止日期：11月24日 23:59
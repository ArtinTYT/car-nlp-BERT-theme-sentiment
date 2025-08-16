# 汽车行业用户观点分析：主题与情感识别（BERT）

## 项目简介

本项目利用 BERT 对汽车行业用户评论数据，进行**主题多标签分类**和**情感三分类**，目标是实现高效、精准的文本自动标签。项目流程包括数据清洗、标签工程、模型设计、训练与评估、可视化等。


## 数据集说明

* **数据目录**：

  ```
  data/
    ├── train.txt
    └── test.txt
  ```
* **数据量**：

    - 训练集：8000 条（df_train.shape = (8000, 6)）
    
    - 测试集：2653 条（df_test.shape = (2653, 6)）

* **格式示例**

  ```
  一直 92 ，偶尔出去不了解当地油品加95(97)。5万公里从没遇到问题，省油，动力也充足，加95也没感觉有啥不同。	油耗#1	动力#1
  ```

  * 每行一条评论，后跟若干“主题#情感”标签（tab分割）。
  * 主题 10 类：动力、价格、内饰、配置、安全性、外观、操控、油耗、空间、舒适性
  * 情感 3 类：正向(1)、中立(0)、负向(-1)


## 1. 数据处理与标签工程

* 自定义 Python 脚本解析 txt 文件，分离文本与主题+情感标签。
* 主题采用**multi-hot编码**，情感采用**sum 规则三分类**（一条评论所有主题情感加和）。
* 标签编码稳定、可追溯，支持多标签和单标签训练任务。

```python
def load_data(file_path): ...
def encode_themes(theme_list): ...
def sum_sentiment_label(senti_list): ...
```

## 2. BERT输入格式和自定义数据集

* 采用`transformers`库的`BertTokenizer`和`BertModel`，自定义`Dataset`类`CarOpinionDataset`，灵活适配单标签和多标签任务。
* 支持多种任务类型（情感三分类、主题多标签分类、主题-情感联合标签）。

```python
class CarOpinionDataset(Dataset):
    ...
    def __getitem__(self, idx):
        ...
        if self.task == 'sentiment':  # 单标签
            label = ...
        elif self.task == 'theme':    # 多标签
            label = ...
        elif self.task == 'theme_sentiments':  # 主题-情感
            label = ...
```


## 3. 模型设计

* **情感三分类**：BERT + Dropout + FC + Softmax
  `nn.CrossEntropyLoss()`
* **主题多标签分类**：BERT + Dropout + FC + Sigmoid
  `nn.BCELoss()`

```python
class BertForSentiment(nn.Module): ...
class BertForMultiLabelTheme(nn.Module): ...
```

## 4. 训练与评估流程

* PyTorch 标准训练/验证循环，指标计算用 sklearn（accuracy, precision, recall, F1，macro 平均）。
* 支持 tqdm 进度条，自动按设备分配数据。
* 自定义训练和评估函数，便于复用和扩展。

```python
def train_one_epoch(model, dataloader, optimizer, criterion): ...
def eval_model(model, dataloader, criterion): ...
def train_multilabel_one_epoch(...): ...
def eval_multilabel_model(...): ...
```


## 5. 可视化

* 使用 matplotlib 绘制 loss/accuracy 曲线，直观展示训练和验证收敛过程。
* 两种任务都支持曲线可视化，便于调参与诊断模型问题。

```python
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
...
```


## 6. 主要实验结果

### 情感三分类（single-label）

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
| ----- | ---------- | -------------- | -------- | ------------ |
| 1     | 0.5469     | 77.58%         | 0.3914   | 86.22%       |
| 2     | 0.4030     | 84.27%         | 0.2196   | 93.66%       |
| 3     | 0.2772     | 89.98%         | 0.1429   | 95.73%       |
| 4     | 0.1817     | 93.83%         | 0.0673   | 98.09%       |
| 5     | 0.1247     | 95.83%         | 0.1422   | 94.81%       |

**分析：**
Loss 稳定下降，准确率不断提升，训练与验证表现接近，模型收敛优异，无明显过拟合。


### 主题多标签分类（multi-label）

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Macro F1 |
| ----- | ---------- | --------- | -------- | ------- | -------- |
| 1     | 0.2350     | 48.56%    | 0.1261   | 77.75%  | 0.8380   |
| 2     | 0.1212     | 76.65%    | 0.0922   | 82.00%  | 0.8764   |
| 3     | 0.0953     | 81.16%    | 0.0761   | 84.11%  | 0.8994   |
| 4     | 0.0807     | 83.31%    | 0.0651   | 85.91%  | 0.9173   |
| 5     | 0.0694     | 85.24%    | 0.0500   | 90.32%  | 0.9418   |

**分析：**
Loss、准确率、Macro F1 曲线健康，最终 Macro F1 可达 0.94+，多标签任务同样泛化良好。


## 7. 项目结构

```
data/
  ├── train.txt
  └── test.txt
model/
  ├── .locks/
  └── models--bert-base-chinese/
main.ipynb             # 全部代码
requirements.txt       # 依赖

models--bert-base-chinese请自己下载添加到文件夹中
```



## 8. 快速使用

1. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```
2. 运行 notebook

   ```bash
   jupyter notebook main.ipynb
   ```
3. 检查可视化结果与模型输出

# car-nlp-BERT-theme-sentiment

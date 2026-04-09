# K210 YOLO-Lite 目标检测训练框架

一个完整的面向 K210 边缘 AI 芯片的 YOLO-Lite 目标检测训练系统。基于 MobileNetV1 骨干网络，支持从数据准备、模型训练到 KModel 格式转换的完整流程。

## 特性

- ✅ 完整的训练 pipeline（数据准备 → 训练 → 模型转换）
- ✅ 一键训练脚本，6 步完成全部流程
- ✅ 基于 MobileNetV1 的轻量级网络，适合边缘设备
- ✅ 支持 K210 专用的 KModel 格式量化转换
- ✅ 修复原版的 TOCO 缺失和 NNCASE 版本兼容问题

## 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Ubuntu 18.04+ / WSL2 |
| Python | **3.7**（必须，TensorFlow 1.13.1 不兼容 3.8+） |
| GPU | NVIDIA GPU（支持 CUDA）推荐 |
| CUDA | 10.0（配合 TensorFlow 1.13.1） |
| cuDNN | 7.4+ |

## 快速开始

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd k210_tra
```

### 2. 创建虚拟环境

```bash
conda create -n k210_train python=3.7 -y
conda activate k210_train
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

验证 TensorFlow 安装：
```bash
python -c "import tensorflow as tf; print(tf.__version__); print('GPU:', tf.test.is_gpu_available())"
```

预期输出：
```
1.13.1
GPU: True
```

### 4. 准备数据集

数据集格式：
```
dataset/<dataset_name>/
├── images/        # 图像文件（JPG/BMP）
├── xml/           # VOC 格式的 XML 标注文件
└── labels.txt     # 类别列表，每行一个类别名
```

示例（`dataset/three_fruit/labels.txt`）：
```
apple
banana
orange
```

### 5. 开始训练

#### 一键训练（推荐）

```bash
# 执行全部 6 个步骤
bash train.sh <dataset_name> all

# 示例
bash train.sh three_fruit all
```

#### 分步训练

```bash
# 仅执行指定步骤
bash train.sh three_fruit 1 2 3 4 5 6

# 例如：仅执行模型转换（第6步）
bash train.sh three_fruit 6
```

## 训练流程详解

| 步骤 | 功能 | 说明 |
|------|------|------|
| 1 | xml2txt | VOC XML 标注转 TXT 格式 |
| 2 | shuffle_txt | 数据集打乱与划分（95%/5%） |
| 3 | generate_anchors | K-Means 聚类生成 YOLO anchor |
| 4 | rewrite_config | 生成训练配置文件（JSON） |
| 5 | training | 模型训练 |
| 6 | model_conversion | 模型转换（ckpt → pb → tflite → kmodel） |

## 输出文件

训练完成后，输出文件位于 `output/results/detection/<dataset_name>/`：

```
output/results/detection/<dataset_name>/
├── det.kmodel      # K210 可用的量化模型 ⭐
├── anchor.txt      # Anchor 参数
├── label.txt       # 类别标签
└── train_loss.txt  # 训练损失记录
```

## 自定义数据集

1. 创建数据集目录：
```bash
mkdir -p dataset/my_dataset/{images,xml}
```

2. 放入图像和对应的 VOC XML 标注文件

3. 创建 `labels.txt`，列出所有类别名：
```bash
cat > dataset/my_dataset/labels.txt << 'EOF'
cat
dog
bird
EOF
```

4. 运行训练：
```bash
bash train.sh my_dataset all
```

## 训练参数调整

编辑 `train.sh` 中的参数：

### Step 4: 配置生成
```bash
python3 rewrite_cfg_json.py \
    -o "${OUTPUT_DIR}/txt/detection/${DATASET_NAME}" \
    -m 100 \              # 最大训练 epoch 数
    -c "${NUM_CLASSES}" \ # 类别数（自动从 labels.txt 读取）
    -d 0.25 \             # 深度乘子（0.125/0.25/0.5）
    -ckpt "${OUTPUT_DIR}/ckpt/detection/${DATASET_NAME}"
```

### Step 5: 模型训练
```bash
python3 train_yolo_lite.py \
    --gpus 0 \            # GPU 设备 ID
    --optimizer adam \    # 优化器（adam/mom/rmsprop/adagrad/adadelta）
    --depth_multiplier 0.25 \  # 网络深度倍增系数
    --bs 8 \              # Batch size
    --lr_policy step      # 学习率策略
```

## 深度乘子选择

| depth_multiplier | 模型大小 | 精度 | 推荐场景 |
|------------------|---------|------|---------|
| 0.125 | 最小 | 较低 | 极端资源受限 |
| 0.25 | 小 | 中等 | **推荐**（平衡精度与速度） |
| 0.50 | 中等 | 较高 | 精度优先 |

## 模型转换流程

```
checkpoint (ckpt)
    ↓
freeze_graph → frozen graph (pb)
    ↓
tflite_convert → TensorFlow Lite (tflite)
    ↓
NNCASE (uint8 量化) → K210 KModel (kmodel)
```

### 修复的问题

原版代码存在以下问题，已在本项目中修复：

1. **TOCO 缺失**：原版使用 `./nncase_tools/toco`，但该文件不存在
   - ✅ 修复：使用 `tflite_convert`（TensorFlow 自带工具）

2. **NNCASE 版本不兼容**：原版 NNCASE 1.0.0 不支持模型中的 Pad 层
   - ✅ 修复：升级到 NNCASE 1.7.0，使用 Python API 进行转换

3. **死循环等待**：原版用 `while` 循环等待文件生成，失败时会卡住
   - ✅ 修复：改为检查命令返回值，失败立即报错

4. **.NET 全球化问题**：NNCASE 需要 ICU 库
   - ✅ 修复：在代码中处理相关环境变量

## 网络架构

```
输入 (1, 240, 320, 3)
    │
    ▼
MobileNetV1 Backbone (depth_multiplier 可配置)
    │  14 层深度可分离卷积
    ▼
Detection Head
    │  3x3 深度可分离卷积
    │  1x1 卷积 (1024 channels)
    │  1x1 卷积 (输出 = anchors_num × (5 + num_classes))
    ▼
输出 (1, 8, 10, anchors_num × (5 + num_classes))
```

- 输入分辨率：320×240（固定）
- 输出特征图：10×8 网格
- 每个网格预测：anchors_num 个边界框
- 每个边界框包含：x, y, w, h, confidence, class_probs

## 项目结构

```
k210_tra/
├── dataset/                          # 数据集目录
│   └── three_fruit/                  # 示例数据集
│       ├── images/                   # 图像文件
│       ├── xml/                      # VOC 格式 XML 标注
│       └── labels.txt                # 类别标签
├── k210_training_code/               # 训练代码
│   └── detection/
│       ├── code/                     # 核心训练代码
│       │   ├── train_yolo_lite.py    # 主训练脚本
│       │   ├── pb2tflite2kmodel.py   # 模型转换脚本 ⭐已修复
│       │   ├── yolo_func.py          # YOLO 核心函数
│       │   ├── base_func.py          # 基础工具函数
│       │   ├── valid_yolo_lite.py    # 验证脚本
│       │   ├── detect_val_images.py  # 检测可视化
│       │   ├── voc_eval.py           # VOC 评估
│       │   ├── models/               # 网络模型定义
│       │   │   └── mobilenetv1_yolo_lite.py
│       │   └── nncase_tools/         # NNCASE 工具（旧版，备用）
│       ├── ckpt_restore/             # 预训练权重
│       │   └── MobileNet_v1_0.25/    # 默认使用
│       └── utils/
│           ├── cfg/cfg.json          # 配置模板
│           └── scripts/              # 数据处理脚本
│               ├── xml2txt.py        # XML 转 TXT
│               ├── shuffle_txt.py    # 数据集划分
│               ├── gen_yolov3_anchor.py  # Anchor 生成
│               ├── rewrite_cfg_json.py   # 配置生成
│               └── kmeans.py         # K-Means 聚类
├── output/                           # 训练输出（自动生成）
│   ├── txt/detection/                # 标签、配置、anchor
│   ├── ckpt/detection/               # 检查点
│   └── results/detection/            # 最终模型结果
├── dataset_process/                  # 数据处理中间文件
├── train.sh                          # 一键训练脚本 ⭐
├── requirements.txt                  # 依赖清单
└── README.md                         # 本文档
```

## 常见问题

### Q: Python 3.8+ 可以运行吗？
**A:** 不可以。TensorFlow 1.13.1 仅支持 Python 3.7，必须使用 Python 3.7。

### Q: 没有 GPU 可以训练吗？
**A:** 可以，但速度会很慢。将 `--gpus 0` 改为 `--gpus ""` 或删除该参数即可使用 CPU。

### Q: 如何中断训练？
**A:** 在数据集目录下创建 `STOP` 文件：
```bash
touch dataset/<dataset_name>/STOP
```

### Q: 模型转换失败怎么办？
**A:** 
1. 确保已安装 nncase 1.7.0：`pip show nncase`
2. 检查验证集图片是否存在：`ls dataset_process/<dataset>/validation/val_images/`
3. 查看转换日志：`cat output/txt/detection/<dataset>/pb2tflite2kmodel_cmd.log`

### Q: 如何部署到 K210？
**A:** 
1. 将 `det.kmodel` 拷贝到 K210 开发板
2. 将 `anchor.txt` 和 `label.txt` 中的参数写入代码
3. 使用 MaixPy 或 C SDK 加载模型进行推理

## 依赖版本说明

本项目关键依赖版本：
- Python: 3.7
- TensorFlow: 1.13.1
- Keras: 2.1.5
- nncase: 1.7.0.20220530

⚠️ 请勿随意升级 TensorFlow 或 nncase 版本，可能不兼容。

## 许可证

本项目基于开源项目修改，遵循 Apache License 2.0。

## 致谢

- 原始项目：基于 Kendryte K210 训练框架
- NNCASE: 神经网络编译器
- TensorFlow: 深度学习框架

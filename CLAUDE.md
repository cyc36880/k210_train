# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

K210 YOLO-Lite 目标检测训练框架，支持从数据准备、模型训练到 KModel 格式转换的完整流程。

## 常用命令

### 一键训练
```bash
# 执行全部步骤
bash train.sh <dataset_name> all

# 执行指定步骤
bash train.sh <dataset_name> 1 2 3 4 5 6
```

### 各步骤说明
1. xml2txt - XML标注转TXT
2. shuffle_txt - 数据集划分
3. generate_anchors - 生成anchor
4. rewrite_config - 生成配置文件
5. training - 模型训练
6. model_conversion - 模型转换

### 环境要求
- Python 3.7（必须）
- TensorFlow 1.13.1
- nncase 1.7.0.20220530

### 关键文件
- `train.sh` - 主训练脚本
- `k210_training_code/detection/code/train_yolo_lite.py` - 训练入口
- `k210_training_code/detection/code/pb2tflite2kmodel.py` - 模型转换（使用nncase 1.7.0 Python API）
- `requirements.txt` - 依赖清单

### 输出位置
- 检查点: `output/ckpt/detection/<dataset>/`
- 最终模型: `output/results/detection/<dataset>/det.kmodel`

### 修复的问题
1. 原TOCO缺失 → 改用tflite_convert
2. NNCASE 1.0.0不兼容 → 升级到1.7.0使用Python API
3. 死循环等待 → 改为检查命令返回值

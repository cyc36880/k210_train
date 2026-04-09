"""基础工具函数模块。

提供训练流程中常用的辅助功能，包括：
- 从文件读取学习率调度表
- 将命令行参数写入文件
- 读取 JSON 格式的超参数配置
- 获取模型检查点文件名
- 加载模型（支持冻结图和检查点两种格式）
"""

from __future__ import absolute_import, division, print_function

import json
import os
import re

import tensorflow as tf
from six import iteritems
from tensorflow.python.platform import gfile


def get_learning_rate_from_file(filename, epoch):
    """从学习率调度文件中读取指定 epoch 对应的学习率。

    文件格式为每行 'epoch:learning_rate'，支持 '#' 注释。
    当 learning_rate 为 '-' 时表示停止训练（返回 -1）。

    参数:
        filename: str，学习率调度文件路径
        epoch: int，当前 epoch 数

    返回:
        float: 对应 epoch 的学习率
    """
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.split("#", 1)[0]  # 去除注释部分
            if line:
                par = line.strip().split(":")
                e = int(par[0])
                if par[1] == "-":
                    lr = -1
                else:
                    lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


def write_arguments_to_file(args, filename):
    """将命令行参数写入文件。

    参数:
        args: argparse.Namespace，命令行参数对象
        filename: str，输出文件路径
    """
    with open(filename, "w") as f:
        for key, value in iteritems(vars(args)):
            f.write("%s: %s\n" % (key, str(value)))


def read_hyperparams_json(path):
    """读取 JSON 格式的超参数配置文件。

    参数:
        path: str，JSON 配置文件路径

    返回:
        dict: 超参数字典
    """
    with open(path) as f:
        params = json.load(f)
    return params


def get_model_filenames(model_dir):
    """获取模型目录中的 meta 文件和 checkpoint 文件名。

    优先使用 TensorFlow 的 checkpoint 状态文件定位 ckpt，
    若不存在则通过文件名匹配最新的 ckpt。

    参数:
        model_dir: str，模型目录路径

    返回:
        tuple: (meta_file, ckpt_file) 文件名

    异常:
        ValueError: 当目录中没有或有多个 meta 文件时抛出
    """
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith(".meta")]
    if len(meta_files) == 0:
        raise ValueError("No meta file found in the model directory (%s)" % model_dir)
    elif len(meta_files) > 1:
        raise ValueError("There should not be more than one meta file in the model directory (%s)" % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    # 在所有 ckpt 文件中查找最大的 step 编号
    meta_files = [s for s in files if ".ckpt" in s]
    max_step = -1
    for f in files:
        step_str = re.match(r"(^model-[\w\- ]+.ckpt-(\d+))", f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def load_model(model, input_map=None):
    """加载 TensorFlow 模型。

    支持两种格式：
    - 冻结图文件（.pb）：直接导入图定义
    - 模型目录（含 .meta 和 .ckpt）：恢复 meta 图和权重

    参数:
        model: str，模型文件路径或模型目录路径
        input_map: dict，可选，输入张量映射
    """
    # 判断是冻结图文件还是包含 meta/ckpt 的目录
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print("Model filename: %s" % model_exp)
        with gfile.FastGFile(model_exp, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name="")
    else:
        print("Model directory: %s" % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print("Metagraph file: %s" % meta_file)
        print("Checkpoint file: %s" % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

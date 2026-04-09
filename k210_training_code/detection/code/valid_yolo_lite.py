"""验证脚本模块。

用于对 YOLO-Lite 模型进行验证集评估，包括：
- 加载已训练的模型（支持 .pb 和检查点格式）
- 对验证集图像进行推理检测
- 应用分数阈值过滤和 NMS 去除冗余框
- 生成 VOC 格式的检测结果文件，用于 AP 评估
"""

from __future__ import absolute_import, division, print_function

import argparse
import sys
from os import environ

import base_func
import numpy as np
import tensorflow as tf
import yolo_func
from keras import backend as K
from PIL import Image


def main(args, params):
    """验证主函数。

    加载模型并对验证集中每张图像进行推理，
    将检测结果按类别写入对应的 txt 文件。

    参数:
        args: argparse.Namespace，命令行参数
        params: dict，超参数配置字典
    """
    # 设置 GPU 设备
    environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # GPU 配置：允许显存按需增长
    gpu_options = tf.GPUOptions(
        allow_growth=True,
        per_process_gpu_memory_fraction=args.gpu_memory_fraction
    )
    sess_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False
    )

    # 解析 anchor 参数
    anchors = params["anchors"]
    anchors = np.array(anchors).reshape(-1, 2)
    anchors = anchors

    # 加载类别信息和阈值参数
    class_names = params["class_names"]
    class_num = params["num_class"]
    score_thresh = args.score_thresh
    iou_thresh = args.iou_thresh

    model_image_size = (params["height"], params["width"])  # 模型输入尺寸

    # 读取验证集标注
    with open(params["valid_set"]) as f:
        annotation_lines = f.readlines()

    # 为每个类别创建检测结果输出文件
    filename = []
    files = []
    for i in range(class_num):
        tmp = "../results/comp4_det_test_" + class_names[i] + ".txt"
        filename.append(tmp)
        tmp_file = open(tmp, "w")
        files.append(tmp_file)

    with tf.Graph().as_default() as graph:
        with tf.Session(config=sess_config) as sess:
            K.set_session(sess)
            # 加载模型
            base_func.load_model(args.model)

            # 获取输入输出张量
            input_image = sess.graph.get_tensor_by_name("inputs:0")
            output = sess.graph.get_tensor_by_name("MobilenetV1/detect_layer/yolo_out/Conv/BiasAdd:0")

            _img_shape = K.placeholder(shape=(2,))  # 原始图像尺寸占位符

            # 构建 YOLO 评估图（包含 NMS）
            _boxes, _scores, _classes = yolo_eval(
                output, anchors, class_num, _img_shape, score_threshold=score_thresh, iou_threshold=iou_thresh
            )

            # 检查模型是否包含 phase_train 节点（用于 batch norm）
            all_tensor_name = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            phase_train_placeholder = None
            if "phase_train" in all_tensor_name:
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # 遍历验证集每张图像进行推理
            for i in range(len(annotation_lines)):
                line = annotation_lines[i].split()
                print(line[0])
                raw_img = Image.open(line[0])
                img_name = line[0].split("/")[-1].split(".")[0]

                input_image_shape = [raw_img.size[1], raw_img.size[0]]  # [h, w]

                if model_image_size != (None, None):
                    # assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
                    # assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'
                    boxed_image = yolo_func.letterbox_image(raw_img, tuple(reversed(model_image_size)))
                else:
                    new_image_size = (raw_img.width - (raw_img.width % 32), raw_img.height - (raw_img.height % 32))
                    boxed_image = yolo_func.letterbox_image(raw_img, new_image_size)

                image_data = np.array(boxed_image, dtype="float32")
                image_data /= 255.0
                image_data = np.expand_dims(image_data, 0)  # 添加批次维度

                # 构建输入字典
                feed_dict = {input_image: image_data, _img_shape: input_image_shape}
                if phase_train_placeholder is not None:
                    feed_dict = {input_image: image_data, _img_shape: input_image_shape, phase_train_placeholder: False}

                # 执行推理
                [out_boxes, out_scores, out_classes] = sess.run([_boxes, _scores, _classes], feed_dict=feed_dict)

                # 将检测结果写入对应类别的文件
                for i, c in reversed(list(enumerate(out_classes))):
                    box = out_boxes[i]
                    score = out_scores[i]

                    top, left, bottom, right = box

                    line = (
                        img_name
                        + " "
                        + str(score)
                        + " "
                        + str(left)
                        + " "
                        + str(top)
                        + " "
                        + str(right)
                        + " "
                        + str(bottom)
                        + "\n"
                    )

                    files[c].write(line)

    # 关闭所有输出文件
    for i in range(class_num):
        files[i].close()


def yolo_eval(yolo_outputs, anchors, num_classes, image_shape, max_boxes=20, score_threshold=0.6, iou_threshold=0.45):
    """评估 YOLO 模型输出，返回过滤后的检测框。

    对网络输出进行解码、分数阈值过滤和非极大值抑制（NMS）。

    参数:
        yolo_outputs: Tensor，网络输出特征图
        anchors: ndarray，anchor 尺寸
        num_classes: int，类别数量
        image_shape: Tensor，原始图像尺寸 [h, w]
        max_boxes: int，每个类别最大检测框数
        score_threshold: float，分数阈值
        iou_threshold: float，NMS 的 IoU 阈值

    返回:
        boxes_: Tensor，过滤后的边界框坐标
        scores_: Tensor，对应的分数
        classes_: Tensor，对应的类别 ID
    """

    input_shape = K.shape(yolo_outputs)[1:3] * 32  # 计算原始输入尺寸（特征图尺寸 * 32）
    grid_shape = yolo_outputs.shape[1:3]
    boxes = []
    box_scores = []

    # 解码网络输出为边界框和分数
    _boxes, _box_scores = yolo_func.yolo_boxes_and_scores(yolo_outputs, anchors, num_classes, input_shape, image_shape)
    boxes.append(_boxes)
    box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    # 应用分数阈值过滤
    mask = box_scores >= score_threshold
    max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
    boxes_ = []
    scores_ = []
    classes_ = []
    # 对每个类别分别执行 NMS
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold
        )
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, "int32") * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def parse_arguments(argv):
    """解析命令行参数。

    参数:
        argv: list，命令行参数列表

    返回:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("cfg_file_path", type=str, help="json cfg file.", default="../training_policy/cfg/cfg.json")
    parser.add_argument(
        "model", type=str, help="Model definition pb file", default="..\camera_demo_K210\pb_file\detect_v1.pb"
    )
    parser.add_argument("--score_thresh", type=float, help="score_threshold.", default=0.005)
    parser.add_argument("--iou_thresh", type=float, help="iou_threshold.", default=0.45)
    parser.add_argument(
        "--gpu_memory_fraction",
        type=float,
        help="Upper bound on the amount of GPU memory that will be used by the process.",
        default=1.0,
    )
    parser.add_argument("--gpus", type=str, help="Indicate the GPUs to be used.", default="-1")

    return parser.parse_args(argv)

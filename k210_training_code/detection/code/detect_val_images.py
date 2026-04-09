"""验证集图像检测与可视化模块。

用于在验证集上运行 YOLO-Lite 模型推理，并在图像上绘制检测结果：
- 加载冻结的 .pb 模型
- 对每张验证图像执行推理
- 绘制检测框、分数和类别名称
- 保存带标注的结果图像
"""

from __future__ import absolute_import, division, print_function

import colorsys
import os.path
from os import environ

import base_func
import numpy as np
import tensorflow as tf
import yolo_func
from keras import backend as K
from PIL import Image, ImageDraw, ImageFont


def detect_images(txt_dir, params):
    """在验证集图像上执行目标检测并保存可视化结果。

    加载已训练的 .pb 模型，对验证集中的每张图像进行检测，
    在原图上绘制检测框和真实框，并保存到结果目录。

    参数:
        txt_dir: str，包含 label.txt 和 shuffle_val_labels.txt 的目录
        params: dict，超参数配置字典
    """
    # 设置使用 CPU 推理
    environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # CPU session 配置（不使用 GPU）
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )

    # 解析 anchor 参数
    anchors = params["anchors"]
    anchors = np.array(anchors).reshape(-1, 2)
    anchors = anchors

    # 读取类别名称和验证集标注
    label_txt = os.path.join(txt_dir, "label.txt")
    with open(label_txt, "r", encoding="utf-8-sig") as f:
        class_names = f.readlines()
    val_txt = os.path.join(txt_dir, "shuffle_val_labels.txt")
    with open(val_txt, "r") as f:
        annotation_lines = f.readlines()
    # 创建检测结果保存目录
    save_dir = os.path.join(txt_dir.replace("\\", "/").replace("txt", "results"), "det_results")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 设置检测参数
    model = os.path.join(params["logs_dir"], "model_final.pb")
    model_image_size = (params["height"], params["width"])  # 模型输入尺寸
    class_num = params["num_class"]
    score_thresh = 0.35    # 分数阈值
    iou_thresh = 0.5       # NMS IoU 阈值
    model_w = 320          # 图像缩放目标宽度
    model_h = 240          # 图像缩放目标高度

    # 为每个类别生成不同的绘制颜色
    hsv_tuples = [(x / class_num, 1.0, 1.0) for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    np.random.seed(10101)   # 固定随机种子保证颜色一致性
    np.random.shuffle(colors)  # 打乱颜色顺序使相邻类别颜色区分明显
    np.random.seed(None)    # 恢复默认随机种子

    with tf.Graph().as_default() as graph:
        with tf.Session(config=sess_config) as sess:
            K.set_session(sess)
            # 加载模型并获取输入输出张量
            base_func.load_model(model)

            input_image = sess.graph.get_tensor_by_name("inputs:0")  # 输入张量
            output = sess.graph.get_tensor_by_name("MobilenetV1/detect_layer/yolo_out/Conv/BiasAdd:0")  # 输出张量

            _img_shape = K.placeholder(shape=(2,))  # 图像尺寸占位符

            # 构建 YOLO 评估图（包含 NMS）
            _boxes, _scores, _classes = yolo_eval(
                output,
                anchors,
                class_num,
                _img_shape,
                score_threshold=score_thresh,
                iou_threshold=iou_thresh,
            )

            # 如果是检查点目录，获取 phase_train 占位符（用于关闭 batch norm）
            if os.path.isdir(model):
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # 遍历验证集每张图像
            for _a, annotation_line in enumerate(annotation_lines):
                line = annotation_line.split()
                test_image = line[0]
                boxes = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])  # 真实框坐标
                raw_img = Image.open(test_image)
                iw, ih = raw_img.size  # 原始图像宽高
                resized_img = raw_img.resize((model_w, model_h), Image.BICUBIC)  # 缩放到模型尺寸

                input_image_shape = [resized_img.size[1], resized_img.size[0]]  # [h, w]

                # letterbox 缩放以保持宽高比
                boxed_image = yolo_func.letterbox_image(resized_img, tuple(reversed(model_image_size)))

                image_data = np.array(boxed_image, dtype="float32")
                image_data /= 255.0
                image_data = np.expand_dims(image_data, 0)  # 添加批次维度

                # 构建 feed_dict
                feed_dict = {input_image: image_data, _img_shape: input_image_shape}
                if os.path.isdir(model):
                    feed_dict = {
                        input_image: image_data,
                        _img_shape: input_image_shape,
                        phase_train_placeholder: False,
                    }

                # 执行推理
                [out_boxes, out_scores, out_classes] = sess.run([_boxes, _scores, _classes], feed_dict=feed_dict)

                # print("Found {} boxes for {}".format(len(out_boxes), "img"))

                # 加载字体用于绘制标签
                font = ImageFont.truetype(
                    font="font/MSYH.TTC",
                    size=np.floor(3e-2 * raw_img.size[1] + 0.5).astype("int32"),
                )
                thickness = 1  # 框线宽度

                # 绘制每个检测结果
                for i, c in reversed(list(enumerate(out_classes))):
                    predicted_class = class_names[c].strip()
                    box = out_boxes[i]
                    score = out_scores[i]

                    label = "{} {:.2f}".format(predicted_class, score)
                    draw = ImageDraw.Draw(raw_img)
                    label_size = draw.textsize(label, font)

                    # 将检测框坐标映射回原始图像尺寸
                    top, left, bottom, right = box
                    top = max(0, np.floor(top / 240 * ih + 0.5).astype("int32"))
                    left = max(0, np.floor(left / 320 * iw + 0.5).astype("int32"))
                    bottom = min(
                        raw_img.size[1],
                        np.floor(bottom / 240 * ih + 0.5).astype("int32"),
                    )
                    right = min(
                        raw_img.size[0],
                        np.floor(right / 320 * iw + 0.5).astype("int32"),
                    )
                    print((left, top), (right, bottom))

                    # 确定标签文本位置
                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    # 绘制检测框和标签背景
                    for i in range(thickness):
                        draw.rectangle(
                            [left + i, top + i, right - i, bottom - i],
                            outline=colors[c],
                        )
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=colors[c],
                    )
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)

                    # 绘制真实标注框（白色）
                    for box in boxes:
                        for i in range(thickness):
                            draw.rectangle(
                                [box[0] + i, box[1] + i, box[2] - i, box[3] - i],
                                outline=(255, 255, 255),
                            )
                    del draw

                # 保存检测结果图像
                raw_img.save(os.path.join(save_dir, "%06d.jpg" % _a))


def yolo_eval(
    yolo_outputs,
    anchors,
    num_classes,
    image_shape,
    max_boxes=20,
    score_threshold=0.6,
    iou_threshold=0.5,
):
    """评估 YOLO 模型输出，返回过滤后的检测框。

    对网络输出进行解码、分数阈值过滤和非极大值抑制（NMS）。

    参数:
        yolo_outputs: Tensor，网络输出特征图
        anchors: ndarray，anchor 尺寸
        num_classes: int，类别数量
        image_shape: Tensor，原始图像尺寸
        max_boxes: int，每个类别最大检测框数
        score_threshold: float，分数阈值
        iou_threshold: float，NMS 的 IoU 阈值

    返回:
        boxes_: Tensor，过滤后的边界框
        scores_: Tensor，对应分数
        classes_: Tensor，对应类别 ID
    """

    input_shape = K.shape(yolo_outputs)[1:3] * 32  # 计算原始输入尺寸
    grid_shape = yolo_outputs.shape[1:3]
    # input_shape = [grid_shape[0]*32,grid_shape[1]*32]
    boxes = []
    box_scores = []

    # 解码边界框和分数
    _boxes, _box_scores = yolo_func.yolo_boxes_and_scores(yolo_outputs, anchors, num_classes, input_shape, image_shape)
    # _boxes = _boxes.eval()
    # _box_scores = _box_scores.eval()
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

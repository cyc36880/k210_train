"""
===============================================================
voc2txt.py
===============================================================

模块功能:
-------------
该模块用于将 VOC 格式的图像数据集转换为训练所需的 TXT 格式标签和归一化 anchor 文件，同时可生成统一编号的 JPG 图片和 XML 注释文件。
主要功能包括：
1. 读取指定目录下的 VOC XML 注释和对应图像。
2. 对图像和注释进行统一编号，并保存到指定输出目录。
3. 将原始图像和 XML 注释文件复制/转换到目标目录。
4. 生成以下文件：
   - labels.txt：每行对应一张图像及其所有目标的坐标和类别 ID。
   - anchors.txt：每个目标的归一化 anchor（中心点 x, y, 宽度，高度）。
   - label.txt：所有类别名称列表。
5. 可选择在图像上显示标注框进行可视化检查。

适用场景:
-------------
- VOC 格式标注数据集的训练数据准备。
- 深度学习目标检测任务（如 YOLO、NanoDet、K210 等）前的数据预处理。
- 需要生成统一编号的图像和 XML 文件用于训练集/验证集划分。

命令行参数说明:
-------------
-i, --img_dir         输入图像目录，默认 "../../../datasets/images/"
-a, --annotation_dir  输入 XML 注释目录，默认 "../../../datasets/xml/"
-sbd, --save_base_dir 输出 TXT 文件存放目录，默认 "../txt/"
-sid, --save_img_dir  输出 JPG 图像目录，默认 "../../../datasets/images_jpg"
-sxd, --save_xml_dir  输出 XML 注释目录，默认 "../../../datasets/xml_jpg"
-sr, --show_rect      是否显示标注框，True/False，默认 False

输出文件说明:
-------------
- ori_all_labels.txt   每行包含图像路径及其所有目标坐标和类别 ID。
- ori_all_anchors.txt  每行包含目标类别 ID 及归一化中心点、宽高。
- label.txt            所有类别名称列表。
- 输出 JPG 图像目录：统一编号保存的图像文件。
- 输出 XML 注释目录：对应编号的 XML 文件副本。

处理流程:
-------------
1. 检查输入图像和 XML 文件数量是否一致。
2. 为每张图像生成统一编号（%06d.jpg / %06d.xml）。
3. 将图像读入，保存为 JPG，复制 XML 文件到输出目录。
4. 遍历 XML 中每个 object：
   - 提取类别名称，构建类别列表。
   - 获取边界框坐标（xmin, ymin, xmax, ymax）。
   - 写入 labels.txt 和 anchors.txt（anchors 归一化）。
5. 如果开启 show_rect 参数，则显示带标注框的图像。
6. 最终保存 label.txt、labels.txt、anchors.txt 并输出日志信息。

使用示例:
-------------
python voc2txt.py \
    -i ../../../datasets/images/ \
    -a ../../../datasets/xml/ \
    -sbd ../txt/ \
    -sid ../../../datasets/images_jpg \
    -sxd ../../../datasets/xml_jpg \
    -sr False
"""


import argparse
import ast
import json
import os
import shutil
import xml.etree.ElementTree as ET
import logging

import cv2
import numpy as np
import tqdm
from scipy import misc


def arg_parser():
    """解析命令行参数。

    返回:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser("voc2txt, code by tom (optimized, python2/3 compatible)")
    parser.add_argument("-i", "--img_dir", type=str, default="../../../datasets/images/", help="input image folder")
    parser.add_argument("-a", "--annotation_dir", type=str, default="../../../datasets/xml/", help="input xml folder")
    parser.add_argument("-sbd", "--save_base_dir", type=str, default="../txt/", help="output folder")
    parser.add_argument("-sid", "--save_img_dir", type=str, default="../../../datasets/images_jpg", help="output jpg folder")
    parser.add_argument("-sxd", "--save_xml_dir", type=str, default="../../../datasets/xml_jpg", help="output xml folder")
    parser.add_argument("-sr", "--show_rect", type=ast.literal_eval, default=False, help="show bounding boxes")
    return parser.parse_args()


def get_files(directory, suffix):
    """获取指定目录下所有指定后缀的文件路径列表。

    参数:
        directory: str，目录路径
        suffix: str，文件后缀（如 '.xml'）

    返回:
        list: 文件绝对路径列表
    """
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files if f.endswith(suffix)]


def get_bgr_image(img_path):
    """读取图像文件（支持中文路径）并返回 BGR 格式。

    使用 np.fromfile + cv2.imdecode 解决中文路径问题。

    参数:
        img_path: str，图像文件路径

    返回:
        ndarray: BGR 格式的图像数据，或 None（读取失败时）
    """
    data = np.fromfile(img_path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def convert(size, box):
    """将像素坐标的边界框转换为归一化的中心点和宽高。

    参数:
        size: tuple (w, h)，图像宽高
        box: tuple (xmin, xmax, ymin, ymax)，像素坐标

    返回:
        tuple: (x_center, y_center, width, height)，均为归一化值 [0, 1]
    """
    w_img, h_img = size
    dw = 1.0 / w_img
    dh = 1.0 / h_img

    xmin, xmax, ymin, ymax = box
    x_center = (xmin + xmax) / 2.0 * dw
    y_center = (ymin + ymax) / 2.0 * dh
    width = (xmax - xmin) * dw
    height = (ymax - ymin) * dh

    return x_center, y_center, width, height


def setup_logger():
    """初始化日志系统。

    返回:
        logging.Logger: 配置好的 logger 实例
    """
    logger = logging.getLogger("voc2txt")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(levelname)s] %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    return logger


def fix_xml_filename(xml_path, new_img_name):
    """
    修复 XML 内部 filename/path 字段
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # <filename>
    name_tag = root.find("filename")
    if name_tag is not None:
        name_tag.text = new_img_name

    # <path>
    path_tag = root.find("path")
    if path_tag is not None:
        path_tag.text = new_img_name

    tree.write(xml_path, encoding="utf-8")


if __name__ == "__main__":
    args = arg_parser()
    log = setup_logger()

    img_dir = args.img_dir
    anno_dir = args.annotation_dir
    save_base_dir = args.save_base_dir
    save_img_dir = args.save_img_dir
    save_xml_dir = args.save_xml_dir
    show_rect = args.show_rect

    # Create output dirs
    for d in [save_base_dir, save_img_dir, save_xml_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            log.info("Created directory: {}".format(d))

    img_files = os.listdir(img_dir)
    anno_files = get_files(anno_dir, ".xml")

    if len(img_files) != len(anno_files):
        raise RuntimeError("Image count must equal annotation count!")

    log.info("Loaded {} images and annotations.".format(len(img_files)))

    save_prefix = "ori_all_"
    anchor_out = open(os.path.join(save_base_dir, "{}anchors.txt".format(save_prefix)), "w")
    label_out = open(os.path.join(save_base_dir, "{}labels.txt".format(save_prefix)), "w")

    classes = []
    img_cnt = 0

    for img_name in tqdm.tqdm(img_files, desc="Processing"):
        img_base = os.path.splitext(img_name)[0]
        anno_path = os.path.join(anno_dir, img_base + ".xml")

        if not os.path.exists(anno_path):
            raise RuntimeError("Annotation missing for {}".format(img_base))

        img_path = os.path.join(img_dir, img_name)
        bgr_img = get_bgr_image(img_path)

        if bgr_img is None:
            log.warning("Image [{}] invalid 3-channel, ignored.".format(img_name))
            continue

        # output name
        new_name = "%06d" % img_cnt
        out_img_path = os.path.join(save_img_dir, new_name + ".jpg")
        out_xml_path = os.path.join(save_xml_dir, new_name + ".xml")

        cv2.imwrite(out_img_path, bgr_img)
        shutil.copy(anno_path, out_xml_path)

        # ★★★ 修复 XML 内 filename/path 字段 ★★★
        fix_xml_filename(out_xml_path, new_name + ".jpg")

        train_img_path = out_img_path.replace("\\", "/")
        label_out.write(train_img_path + " ")

        tree = ET.parse(out_xml_path)
        root = tree.getroot()

        h, w = bgr_img.shape[:2]

        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in classes:
                classes.append(cls)

            cls_id = classes.index(cls)

            xmlbox = obj.find("bndbox")
            xmin = float(xmlbox.find("xmin").text)
            xmax = float(xmlbox.find("xmax").text)
            ymin = float(xmlbox.find("ymin").text)
            ymax = float(xmlbox.find("ymax").text)

            # label
            label_out.write("{},{},{},{},{} ".format(
                int(xmin), int(ymin), int(xmax), int(ymax), cls_id))

            # normalized anchor
            norm = convert((w, h), (xmin, xmax, ymin, ymax))
            anchor_out.write("{} {}\n".format(cls_id, " ".join([str(v) for v in norm])))

            if show_rect:
                cv2.rectangle(bgr_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 255), 2)

        label_out.write("\n")

        if show_rect:
            cv2.imshow("annotation", bgr_img)
            cv2.waitKey(0)

        img_cnt += 1

    anchor_out.close()
    label_out.close()

    # class names
    label_file = os.path.join(save_base_dir, "label.txt")
    with open(label_file, "w", encoding="utf-8-sig") as f:
        for cls in classes:
            f.write(cls + "\n")

    log.info("Finished. Total valid images: {}".format(img_cnt))
    log.info("Classes: {}".format(classes))
    log.info("Output saved under: {}".format(save_base_dir))

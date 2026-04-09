# coding=utf-8
"""
anchor_generator.py

说明:
该脚本用于根据训练标签生成YOLO模型所需的anchor框。
通过读取标注txt文件中的宽高信息，使用k-means聚类生成指定数量的anchor，
并输出到anchor.txt文件中。脚本兼容Python2/3，不使用f-string。

功能:
1. 读取训练标签txt文件，获取宽高信息。
2. 使用k-means聚类计算anchor中心。
3. 输出anchor的像素坐标和相对坐标到文本文件。
"""

import argparse
import os
import logging
import numpy as np
from kmeans import kmeans


def arg_parser():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser("Generate YOLO anchors using k-means clustering")
    parser.add_argument(
        "-i", "--input_txt_dir", type=str, default="../txt",
        help="Folder directory containing all label txt files"
    )
    parser.add_argument(
        "-n", "--anchor_num", type=int, default=9,
        help="Number of anchors to generate"
    )
    parser.add_argument(
        "-tw", "--train_width", type=int, default=320,
        help="Training image width"
    )
    parser.add_argument(
        "-th", "--train_height", type=int, default=240,
        help="Training image height"
    )
    return parser.parse_args()


def compute_centroids(label_path, n_anchors, img_h, img_w, anchor_out_path):
    """
    根据label文件计算anchor中心，并写入输出文件

    参数:
        label_path: 标注文件路径
        n_anchors: 需要生成的anchor数量
        img_h: 训练图片高度
        img_w: 训练图片宽度
        anchor_out_path: 输出anchor的文件路径
    """
    logging.info("读取标注文件: %s", label_path)
    dataset = []

    # 读取label txt文件，每行格式为: class x_center y_center width height
    with open(label_path, 'r') as f:
        for line in f:
            temp = line.strip().split(" ")
            if len(temp) > 1:
                # 获取宽高信息（归一化的值）
                dataset.append([float(temp[3]), float(temp[4])])

    if not dataset:
        logging.warning("没有读取到有效的标签数据!")
        return

    logging.info("运行k-means聚类生成 %d 个anchors", n_anchors)
    # 运行k-means聚类
    anchors = kmeans(np.array(dataset), k=n_anchors)

    # 将归一化坐标转换为像素值
    coord = [(anchors[i, 0] * img_w, anchors[i, 1] * img_h) for i in range(n_anchors)]
    # 按面积排序
    coord.sort(key=lambda x: x[0] * x[1])

    logging.info("Anchor计算结果（像素坐标）:")
    for w, h in coord:
        logging.info("%d, %d", int(w), int(h))

    # 写入anchor.txt
    with open(anchor_out_path, 'w') as fid:
        # 输出像素坐标
        for i in range(len(coord)):
            if i == len(coord) - 1:
                fid.write("%.2f, %.2f\n" % (coord[i][0], coord[i][1]))
            else:
                fid.write("%.2f, %.2f, " % (coord[i][0], coord[i][1]))

        # 输出相对坐标（/32）
        for i in range(len(coord)):
            if i == len(coord) - 1:
                fid.write("%.2f, %.2f\n" % (coord[i][0] / 32, coord[i][1] / 32))
            else:
                fid.write("%.2f, %.2f, " % (coord[i][0] / 32, coord[i][1] / 32))

    logging.info("Anchor已保存到: %s", anchor_out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    args = arg_parser()

    input_txt_dir = args.input_txt_dir
    anchor_path = os.path.join(input_txt_dir, "shuffle_anchors.txt")
    anchor_out_path = os.path.join(input_txt_dir, "anchor.txt")

    compute_centroids(anchor_path, args.anchor_num, args.train_height, args.train_width, anchor_out_path)

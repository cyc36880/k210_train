"""
===============================================================
shuffle_txt.py
===============================================================

模块功能:
-------------
该模块用于对已有的目标检测数据集进行训练/验证集划分，并生成对应的标签和验证集文件。
主要功能包括：
1. 从指定目录读取原始 labels.txt 和 anchors.txt。
2. 随机打乱样本顺序。
3. 根据指定训练比例划分训练集和验证集。
4. 将验证集的 XML 注释文件和对应的 JPG 图像复制到指定目录。
5. 生成 train_labels.txt、val_labels.txt、val.txt 和 anchors.txt。

适用场景:
-------------
- VOC 格式标注数据集的训练/验证集划分。
- 需要生成 K210 或其他深度学习训练环境可用的标签文件。
- 数据集较大时，支持随机抽样和 batch_size 补充逻辑。

命令行参数说明:
-------------
-i, --input_txt_dir      原始 labels 和 anchors 的存放目录，默认 "../txt"
-a, --annotation_dir     XML 标注文件目录，默认 "../../../datasets/xml/"
-t, --train_img_dir      JPG 图片目录，默认 "../../../datasets/images_jpg"
-s, --save_val_dir       验证集输出目录，默认 "../../../datasets/validation"
-tr, --train_ratio       训练集比例，默认 0.95
-bs, --batch_size        最小训练/验证集数量补充，默认 32

输出文件说明:
-------------
- train_labels.txt        打乱后训练集的 labels 文件
- val_labels.txt          打乱后验证集的 labels 文件
- val.txt                 验证集文件名列表（对应 val_images 中的文件）
- anchors.txt             拷贝自原始 anchors.txt，用于训练
- val_images/             验证集图片目录
- val_xml/                验证集 XML 注释目录

流程概述:
-------------
1. 读取原始 labels 和 anchors。
2. 获取 annotation_dir 下的所有 XML 文件。
3. 随机打乱样本索引。
4. 根据 train_ratio 划分训练集和验证集。
5. 对验证集：
   - 复制 XML 文件到 val_xml/
   - 复制对应 JPG 文件到 val_images/
   - 生成 val_labels.txt 和 val.txt
6. 对训练集：
   - 生成 train_labels.txt
7. 如果训练集或验证集数量不足 batch_size，则随机补充。
8. 输出日志信息，显示数据划分情况。

使用示例:
-------------
python shuffle_txt.py \
    -i ../txt \
    -a ../../../datasets/xml/ \
    -t ../../../datasets/images_jpg \
    -s ../../../datasets/validation \
    -tr 0.95 \
    -bs 32
"""


import argparse
import os
from shutil import copyfile
import numpy as np
import logging

# ---------------------------------------------------------------
# 初始化日志系统（不改变程序行为，只输出信息）
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)


def arg_parser():
    """命令行参数解析"""
    parser = argparse.ArgumentParser("shuffle txt and gen train/val labels, code by tom")
    parser.add_argument("-i", "--input_txt_dir", type=str, default="../txt", help="folder dir to all labels and anchors")
    parser.add_argument("-a", "--annotation_dir", type=str, default="../../../datasets/xml/", help="folder dir to input xml")
    parser.add_argument("-t", "--train_img_dir", type=str, default="../../../datasets/images_jpg", help="folder dir to output jpg images")
    parser.add_argument("-s", "--save_val_dir", type=str, default="../../../datasets/validation", help="folder dir to validation")
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.95, help="train ratio")
    parser.add_argument("-bs", "--batch_size", default=32, type=int, help="training_batch_size")
    return parser.parse_args()


def getFiles(file_dir, suffix):
    """获取指定目录下所有指定后缀的文件列表"""
    result = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith(suffix):
                result.append(os.path.join(root, file))
    return result


def copy_ann_and_img(index, annotation_files, annotation_suffix,
                     output_val_xml_dir, train_img_dir,
                     img_suffix, output_val_img_dir, save_base_prefix):
    """复制指定索引的 XML 标注文件和对应的图像到验证集目录。

    参数:
        index: int，样本索引
        annotation_files: list，XML 标注文件路径列表
        annotation_suffix: str，标注文件后缀（如 '.xml'）
        output_val_xml_dir: str，验证集 XML 输出目录
        train_img_dir: str，训练图像目录
        img_suffix: str，图像文件后缀（如 '.jpg'）
        output_val_img_dir: str，验证集图像输出目录
        save_base_prefix: str，输出文件名前缀
    """

    # 生成输出文件名（兼容旧 Python 版本）
    dst_xml_name = "%s%08d%s" % (save_base_prefix, index, annotation_suffix)
    dst_img_name = "%s%08d%s" % (save_base_prefix, index, img_suffix)

    src_xml_file = annotation_files[index]
    dst_xml_file = os.path.join(output_val_xml_dir, dst_xml_name)
    copyfile(src_xml_file, dst_xml_file)

    # 对应 image
    anno_file_name = os.path.splitext(os.path.basename(src_xml_file))[0]
    src_img_file = os.path.join(train_img_dir, anno_file_name + img_suffix)
    dst_img_file = os.path.join(output_val_img_dir, dst_img_name)
    copyfile(src_img_file, dst_img_file)


if __name__ == "__main__":
    args = arg_parser()

    logging.info("读取输入目录和基础配置...")

    input_txt_dir = args.input_txt_dir
    input_labels_txt = os.path.join(input_txt_dir, "ori_all_labels.txt")
    input_anchors_txt = os.path.join(input_txt_dir, "ori_all_anchors.txt")

    annotation_files = getFiles(args.annotation_dir, ".xml")

    train_img_dir = args.train_img_dir
    save_val_dir = args.save_val_dir
    if not os.path.exists(save_val_dir):
        os.makedirs(save_val_dir)

    save_base_prefix = "shuffle_"
    img_suffix = ".jpg"
    annotation_suffix = ".xml"
    batch_size = args.batch_size

    logging.info("准备验证集输出子目录...")
    output_val_img_dir = os.path.join(save_val_dir, "val_images")
    output_val_xml_dir = os.path.join(save_val_dir, "val_xml")

    # 清空或创建 val_images / val_xml
    for d in [output_val_img_dir, output_val_xml_dir]:
        if not os.path.exists(d):
            os.mkdir(d)
        else:
            suffix = ".xml" if "xml" in d else img_suffix
            old_files = getFiles(d, suffix)
            for file in old_files:
                os.remove(file)

    output_test_txt = os.path.join(save_val_dir, "val.txt")
    output_train_txt = os.path.join(input_txt_dir, save_base_prefix + "train_labels.txt")
    output_val_txt = os.path.join(input_txt_dir, save_base_prefix + "val_labels.txt")
    output_anchors_txt = os.path.join(input_txt_dir, save_base_prefix + "anchors.txt")

    logging.info("复制 anchors 文件...")
    copyfile(input_anchors_txt, output_anchors_txt)

    logging.info("读取 labels 列表...")
    with open(input_labels_txt) as f:
        labels = f.readlines()

    total_num = len(labels)
    logging.info("标签总数量: %d" % total_num)

    seq_list = np.random.permutation(total_num)
    train_num = int(args.train_ratio * total_num)
    val_num = total_num - train_num

    logging.info("训练集数量: %d, 验证集数量: %d" % (train_num, val_num))

    f_train = open(output_train_txt, "w")
    f_val = open(output_val_txt, "w")
    f_test = open(output_test_txt, "w")

    logging.info("开始分配训练 / 验证样本...")

    for i, index in enumerate(seq_list):
        label = labels[index].replace("\\", "/")

        if i < train_num:
            f_train.write(label)
        else:
            copy_ann_and_img(index, annotation_files, annotation_suffix,
                             output_val_xml_dir, train_img_dir, img_suffix,
                             output_val_img_dir, save_base_prefix)

            f_val.write(label)
            f_test.write("%s%08d\n" % (save_base_prefix, index))

    logging.info("检查 train/val 是否达到 batch_size...")

    while train_num < batch_size:
        index = np.random.randint(total_num)
        f_train.write(labels[index].replace("\\", "/"))
        train_num += 1
        logging.info("补充 train，当前数量: %d" % train_num)

    while val_num < batch_size:
        index = np.random.randint(total_num)

        copy_ann_and_img(index, annotation_files, annotation_suffix,
                         output_val_xml_dir, train_img_dir, img_suffix,
                         output_val_img_dir, save_base_prefix)

        f_val.write(labels[index].replace("\\", "/"))
        f_test.write("%s%08d\n" % (save_base_prefix, index))
        val_num += 1
        logging.info("补充 val，当前数量: %d" % val_num)

    f_train.close()
    f_val.close()
    f_test.close()

    logging.info("数据打乱 + 训练/验证集生成完成！")

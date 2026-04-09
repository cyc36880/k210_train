"""模型转换模块：将 TensorFlow 检查点转换为 K210 可用的 KModel 格式。

流程：
1. 加载 TensorFlow 检查点并冻结图为 .pb 文件
2. 在验证集上运行检测并保存可视化结果
3. 使用 TOCO 将 .pb 转换为 .tflite 格式
4. 使用 NNCASE 将 .tflite 转换为 K210 的 .kmodel 格式
"""

import argparse
import importlib
import logging
import os
import shutil
import sys
from os import environ

import base_func
import cv2
import numpy as np
import tensorflow as tf
from detect_val_images import detect_images
from tensorflow.python.framework import graph_util

def setup_logger(log_path=None):
    """设置日志系统，支持控制台和文件输出。

    参数:
        log_path: str，可选，日志文件保存路径
    """
    # 创建或获取 root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建 formatter
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_path:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

CANMV_LOG_PATH = os.getenv("CANMV_LOG_PATH")
setup_logger(CANMV_LOG_PATH)


def freeze_graph(params_path, ckpt_foler_name, kmodel_save_dir, coord_dir, max_epoches):
    """冻结 TensorFlow 检查点并转换为 KModel 格式。

    完整流程：加载检查点 -> 冻结图 -> 检测可视化 -> pb转tflite -> tflite转kmodel

    参数:
        params_path: str，JSON 配置文件路径
        ckpt_foler_name: str，检查点文件夹名称
        kmodel_save_dir: str，kmodel 保存目录
        coord_dir: str，量化数据集目录
        max_epoches: int，训练总轮数（用于损失输出）
    """
    # 使用 CPU 执行转换
    environ["CUDA_VISIBLE_DEVICES"] = "-1"
    params = base_func.read_hyperparams_json(params_path)

    # 创建 kmodel 保存目录
    if not os.path.exists(kmodel_save_dir):
        os.mkdir(kmodel_save_dir)
    cfg_name = "cfg_" + params["logs_dir"].split("_")[-1] + ".json"
    # 复制 anchor 和 label 文件到输出目录
    anchor_in_txt = params_path.replace(cfg_name, "anchor.txt")
    label_in_txt = params_path.replace(cfg_name, "label.txt")
    anchor_out_txt = os.path.join(kmodel_save_dir, "anchor.txt")
    label_out_txt = os.path.join(kmodel_save_dir, "label.txt")
    # Copy other required files
    if os.path.exists(anchor_in_txt):
        shutil.copy(anchor_in_txt, anchor_out_txt)
    if os.path.exists(label_in_txt):
        shutil.copy(label_in_txt, label_out_txt)
    # 从配置中解析深度乘子
    depth_multiplier = int(params["logs_dir"].split("_")[-1]) / 1000
    input_dir = params["logs_dir"]
    
    # 检查检查点文件是否存在
    if not os.path.exists(input_dir):
        raise ValueError(f"训练输出目录不存在: {input_dir}\n请先完成Step 5训练步骤")
    
    # 获取 meta 和 ckpt 文件名
    meta_file, ckpt_file = base_func.get_model_filenames(input_dir)

    # 导入网络定义并构建推理图
    model_def = "models.mobilenetv1_yolo_lite"
    network = importlib.import_module(model_def)
    images_placeholder = tf.placeholder(tf.float32, shape=(1, 240, 320, 3), name="inputs")  # 输入占位符，固定 240x320
    # 解析 anchor 参数
    classes = params["num_class"]
    anchors = params["anchors"]
    anchors = np.array(anchors).reshape(-1, 2)
    anchors_num = len(anchors)
    # 构建网络推理图
    logits, _ = network.inference(
        images_placeholder,
        phase_train=False,
        yolo_conv_depth=(4 + 1 + classes) * anchors_num,
        depth_multiplier=depth_multiplier,
    )
    # 设置输出文件路径
    ckpt_dir_exp = os.path.expanduser(input_dir)
    output_node_names = "MobilenetV1/detect_layer/yolo_out/Conv/BiasAdd"  # 输出节点名
    output_graph = os.path.join(input_dir, "model_final.pb")         # 冻结图输出路径
    frozen_tflite_file = os.path.join(input_dir, "model_final.tflite")  # TFLite 输出路径
    frozen_kmodel_file = os.path.join(kmodel_save_dir, "det.kmodel")    # KModel 输出路径
    meta_file = os.path.join(ckpt_dir_exp, meta_file)
    ckpt_file = os.path.join(ckpt_dir_exp, ckpt_file)
    logging.info("meta-file is %s" % meta_file)
    logging.info("ckpt-file is %s" % ckpt_file)
    saver = tf.train.Saver(tf.global_variables())
    graph = tf.get_default_graph()            # 获取默认图
    input_graph_def = graph.as_graph_def()     # 获取图的序列化定义
    with tf.Session() as sess:
        saver.restore(sess, ckpt_file)         # 恢复检查点权重

        # 修复 batch norm 相关节点（RefSwitch -> Switch, AssignSub -> Sub 等）
        for node in input_graph_def.node:
            if node.op == "RefSwitch":
                node.op = "Switch"
                for index in range(len(node.input)):
                    if "moving_" in node.input[index]:
                        node.input[index] = node.input[index] + "/read"
            elif node.op == "AssignSub":
                node.op = "Sub"
                if "use_locking" in node.attr:
                    del node.attr["use_locking"]
            elif node.op == "AssignAdd":
                node.op = "Add"
                if "use_locking" in node.attr:
                    del node.attr["use_locking"]

        # 将变量转换为常量，生成冻结图
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(","),
        )  # 如果有多个输出节点，以逗号隔开
        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存冻结图
            f.write(output_graph_def.SerializeToString())  # 序列化并写入

        # 在验证集上运行检测并保存可视化结果
        detect_images(params_path.replace(cfg_name, ""), params)

        # 将训练损失缩放到指定 epoch 数并保存
        num_epoch = max_epoches

        train_loss_in_file = params_path.replace(cfg_name, "train_loss.txt")
        with open(train_loss_in_file, "r") as f:
            train_loss_str = f.readlines()
            train_loss_ = np.array([float(x) for x in train_loss_str])
        train_loss = cv2.resize(train_loss_, (1, num_epoch))
        train_loss_out_file = os.path.join(kmodel_save_dir, "train_loss.txt")
        with open(train_loss_out_file, "w") as f:
            for _i in range(num_epoch):
                f.write("%f\n" % train_loss[_i][0])
        logging.info(
            "%d ops in the final graph." % len(output_graph_def.node)
        )
        # 将命令行日志输出到 txt 目录
        txt_dir = os.path.dirname(params_path)
        cmd_log_path = os.path.join(txt_dir, "pb2tflite2kmodel_cmd.log")
        # 执行 tflite_convert 将 pb 转换为 tflite
        toco_cmd = (
            "tflite_convert --graph_def_file=%s --output_file=%s --input_arrays=inputs --output_arrays=%s --input_shapes=1,240,320,3 --inference_type=FLOAT >> %s 2>&1"
            % (output_graph, frozen_tflite_file, output_node_names, cmd_log_path)
        )

        logging.info("执行 tflite_convert 转换...")
        ret = os.system(toco_cmd)  # 执行 tflite 转换
        if ret != 0:
            logging.error("tflite_convert 转换失败，请检查 %s" % cmd_log_path)
            raise RuntimeError("pb 转 tflite 失败")
        if not os.path.exists(frozen_tflite_file):
            logging.error("tflite 文件未生成: %s" % frozen_tflite_file)
            raise RuntimeError("tflite 文件未生成")
        # 执行 NNCASE 将 tflite 转换为 kmodel（uint8 量化）
        # 使用 nncase 1.7.0 Python API
        logging.info("执行 NNCASE 转换 (使用 nncase 1.7.0 Python API)...")
        try:
            import nncase
            # 读取 tflite 模型
            with open(frozen_tflite_file, 'rb') as f:
                tflite_model = f.read()

            # 配置编译选项
            compile_options = nncase.CompileOptions()
            compile_options.target = "k210"
            compile_options.input_type = "float32"
            compile_options.input_shape = [1, 240, 320, 3]

            # 创建编译器并导入模型
            compiler = nncase.Compiler(compile_options)
            import_options = nncase.ImportOptions()
            compiler.import_tflite(tflite_model, import_options)

            # 配置 PTQ 量化选项（使用验证集图片）
            ptq_options = nncase.PTQTensorOptions()
            ptq_options.samples_count = 10  # 使用10张图片进行量化校准

            # 从验证集目录加载校准图片
            calib_images = []
            for img_name in os.listdir(coord_dir)[:ptq_options.samples_count]:
                if img_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(coord_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (320, 240))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        calib_images.append(img)

            if len(calib_images) > 0:
                # 创建校准数据张量
                calib_data = np.stack(calib_images).astype(np.float32)
                ptq_options.set_tensor_data(calib_data.tobytes())
                compiler.use_ptq(ptq_options)

            # 编译模型
            compiler.compile()

            # 生成 kmodel
            kmodel_bytes = compiler.gencode_tobytes()
            with open(frozen_kmodel_file, 'wb') as f:
                f.write(kmodel_bytes)

            logging.info("NNCASE 转换成功: %s" % frozen_kmodel_file)
        except Exception as e:
            logging.error("NNCASE 转换失败: %s" % str(e))
            raise RuntimeError("tflite 转 kmodel 失败: %s" % str(e))


def main(args):
    """主函数，调用 freeze_graph 执行模型转换。

    参数:
        args: argparse.Namespace，命令行参数
    """
    freeze_graph(
        args.cfg_file_path,
        args.ckpt_folder_name,
        args.kmodel_dir,
        args.coord_dir,
        args.max_epoches,
    )


def parse_arguments(argv):
    """解析命令行参数。

    参数:
        argv: list，命令行参数列表

    返回:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ckpt_folder_name",
        type=str,
        help="name of folder containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters",
    )
    parser.add_argument(
        "cfg_file_path",
        type=str,
        help="json cfg file.",
        default="cfg/widerface_cfg.json",
    )
    parser.add_argument("kmodel_dir", default=r"../kmodel/", type=str, help="folder dir to kmodel")
    parser.add_argument(
        "coord_dir",
        default=r"../coord/",
        type=str,
        help="folder dir to coordinate datasets",
    )
    parser.add_argument("-m", "--max_epoches", default=800, type=int, help="training epoch number")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

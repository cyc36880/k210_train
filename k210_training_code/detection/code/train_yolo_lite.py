"""
train_yolo.py

功能说明:
1. 根据指定的配置文件训练 YOLO 网络。
2. 支持多种学习率策略和优化器。
3. 可加载预训练模型，并选择是否保留检测层参数。
4. 自动创建日志目录并记录训练日志。
5. 支持保存模型权重和 meta 图。
6. 打印训练过程详细信息，包括每 batch 的 loss 和学习率。

注意:
- logging 不是多进程安全的，同时写入同一日志文件可能会丢失日志。
- 不使用 f-string，保证 Python 2/3 兼容。
"""

from __future__ import absolute_import, division, print_function

import argparse
import ast
import importlib
import logging
import os
import random
import sys
import time
from datetime import datetime
from os import environ
import base_func
import numpy as np
import tensorflow as tf
import yolo_func
from keras import backend as K


def setup_logger(log_path=None):
    """设置日志系统，支持控制台输出和文件输出"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件日志
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


CANMV_LOG_PATH = os.getenv("CANMV_LOG_PATH")
setup_logger(CANMV_LOG_PATH)


def save_variables_and_metagraph(sess, saver, summary_writer, log_dir, model_name, step):
    """保存模型权重和 meta 图，并记录保存耗时"""
    logging.info("开始保存模型参数")
    start_time = time.time()
    checkpoint_path = os.path.join(log_dir, "model-%s.ckpt" % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    logging.info("模型参数保存完成, 耗时 %.2f 秒" % save_time_variables)

    metagraph_filename = os.path.join(log_dir, "model-%s.meta" % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        logging.info("开始保存 meta 图")
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        logging.info("Meta 图保存完成, 耗时 %.2f 秒" % save_time_metagraph)

    # 写入 TensorBoard
    summary = tf.Summary()
    summary.value.add(tag="time/save_variables", simple_value=save_time_variables)
    summary.value.add(tag="time/save_metagraph", simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def main(args, params):
    """主训练函数"""
    # 设置 GPU 设备
    environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # GPU 配置
    gpu_options = tf.GPUOptions(
        allow_growth=True,
        per_process_gpu_memory_fraction=args.gpu_memory_fraction
    )
    sess_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False
    )
    # 启用 XLA JIT 编译加速
    sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    # 设置 Keras 后端 session
    K.set_session(tf.Session(config=sess_config))
    logging.info("GPU 加速已启用 (allow_growth=True, memory_fraction=%.2f, XLA=ON)" % args.gpu_memory_fraction)

    # 构建 STOP 和 DELETE 文件路径，用于训练中断或删除
    cfg_name = "cfg_" + params["logs_dir"].split("_")[-1] + ".json"
    stop_path = args.cfg_file_path.replace("\\", "/").replace(
        "output/txt/detection", "datasets/detection").replace(cfg_name, "STOP")
    delete_path = args.cfg_file_path.replace("\\", "/").replace(
        "output/txt/detection", "datasets/detection").replace(cfg_name, "DELETE")
    loss_txt = args.cfg_file_path.replace(cfg_name, "train_loss.txt")

    # 导入网络定义
    model_def = params["model_def"]
    network = importlib.import_module(model_def)

    input_shape = (params["height"], params["width"])
    grid_shape = (params["grid_h"], params["grid_w"])

    anchors = np.array(params["anchors"]).reshape(-1, 2)
    anchors_num = len(anchors)

    # 创建日志目录
    subdir = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
    log_dir = os.path.expanduser(params["logs_dir"])
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.info("日志目录: %s" % log_dir)

    # 保存训练参数到文件
    base_func.write_arguments_to_file(args, os.path.join(log_dir, "arguments.txt"))

    # 设置随机种子
    np.random.seed(seed=int(time.time()))
    random.seed(int(time.time()) + 10)

    pretrained_model = args.pretrained_model
    if pretrained_model:
        pretrained_model = os.path.expanduser(pretrained_model)
        logging.info("使用预训练模型: %s" % pretrained_model)

    # 读取训练集
    with open(params["train_set"]) as f:
        annotation_lines = f.readlines()
    epoch_size = len(annotation_lines) // args.bs

    num_classes = params["num_class"]
    depth_multiplier = args.depth_multiplier
    _op_ = args.optimizer.lower()
    lr_policy = args.lr_policy.lower()

    with tf.Graph().as_default():
        tf.set_random_seed(int(time.time()) + 100)

        # 学习率变量
        init_lr = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="init_learning_rate")
        run_lr = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="running_learning_rate")
        num_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name="num_epoch")

        # 构建学习率策略
        learning_rate = None
        if lr_policy == "step":
            pass
        elif lr_policy == "exp":
            learning_rate = tf.train.exponential_decay(init_lr, num_epoch, lrp.decay_steps, lrp.decay_rate, staircase=lrp.staircase)
        elif lr_policy == "natural_exp":
            learning_rate = tf.train.natural_exp_decay(init_lr, num_epoch, lrp.decay_steps, lrp.decay_rate, staircase=lrp.staircase)
        elif lr_policy == "poly":
            learning_rate = tf.train.polynomial_decay(init_lr, num_epoch, lrp.decay_steps, lrp.end_learning_rate, lrp.power, cycle=lrp.cycle)
        elif lr_policy == "cos":
            learning_rate = tf.train.cosine_decay(init_lr, num_epoch, lrp.decay_steps, alpha=lrp.alpha)
        elif lr_policy == "inv":
            learning_rate = tf.train.inverse_time_decay(init_lr, num_epoch, lrp.decay_steps, lrp.decay_rate, staircase=lrp.staircase)
        else:
            raise AssertionError("不支持的学习率策略: %s" % lr_policy)

        # 占位符
        phase_train_placeholder = tf.placeholder(tf.bool, name="phase_train")
        labels = tf.placeholder(tf.float32, shape=(None, grid_shape[0], grid_shape[1], anchors_num, 5 + num_classes), name="label")
        inputs = tf.placeholder(tf.float32, shape=(None, input_shape[0], input_shape[1], 3), name="inputs")

        logging.info("构建网络图...")
        y_pred, _ = network.inference(inputs, phase_train=phase_train_placeholder,
                                      yolo_conv_depth=(5 + num_classes) * anchors_num,
                                      weight_decay=params["weight_decay"],
                                      depth_multiplier=depth_multiplier)
        yolo_loss = yolo_func.yolo_loss(input_shape, grid_shape, y_pred, labels, anchors, num_classes, params["ignore_thresh"], print_loss=False)

        global_step = tf.train.get_or_create_global_step()

        # 构建 Saver
        var_list = list(set(tf.trainable_variables() + [v for v in tf.global_variables() if "moving_mean" in v.name or "moving_variance" in v.name]))
        saver = tf.train.Saver(var_list=var_list, max_to_keep=20)

        # 预训练模型加载
        if pretrained_model:
            if args.pretrain_with_det:
                exclude_var = []
            else:
                exclude_var = [v for v in var_list if "Logits" in v.name or "detect_layer" in v.name or "gamma" in v.name]
            saver_restore = tf.train.Saver(var_list=list(set(var_list) - set(exclude_var)))

        # 选择优化器
        if _op_ == "adagrad":
            optimizer = tf.train.AdagradOptimizer(run_lr)
        elif _op_ == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(run_lr, rho=0.9, epsilon=1e-6)
        elif _op_ == "adam":
            optimizer = tf.train.AdamOptimizer(run_lr, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif _op_ == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(run_lr, decay=0.9, momentum=0.9, epsilon=1.0)
        elif _op_ == "mom":
            optimizer = tf.train.MomentumOptimizer(run_lr, 0.9, use_nesterov=True)
        else:
            raise ValueError("无效优化器: %s" % _op_)

        # 是否使用 batch norm
        if params["use_batch_norm"]:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(yolo_loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(yolo_loss, global_step=global_step)

        # 创建 session (使用 GPU 加速配置)
        sess = tf.Session(config=sess_config)
        K.set_session(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        batch_size = args.bs
        max_epochs = params["max_epochs"]
        lr_set = params["learning_rate"]
        run_lr_all = []

        # 加载预训练模型
        if pretrained_model:
            logging.info("正在恢复预训练模型: %s" % pretrained_model)
            if os.path.isdir(pretrained_model):
                saver_restore.restore(sess, tf.train.latest_checkpoint(pretrained_model))
            else:
                saver_restore.restore(sess, pretrained_model)

        # 开始训练
        logging.info("开始训练...")
        batch_number = 0
        np.random.shuffle(annotation_lines)
        annotation_lines = annotation_lines[0: epoch_size * batch_size]
        avg_loss = 0.0
        # print(grid_shape)
        for data_pack, epoch in yolo_func.data_generator_wrapper(annotation_lines, batch_size, max_epochs, input_shape,grid_shape, anchors, num_classes):
            img_data = data_pack[0]
            y_true = data_pack[1]

            epoch_tmp = epoch
            if (batch_number + 1) % epoch_size == 0:
                epoch_tmp = epoch - 1

            # 第一个 batch 初始化学习率
            if batch_number == 0:
                if lr_set > 0.0:
                    lr = lr_set
                else:
                    lr = base_func.get_learning_rate_from_file(params["learning_rate_schedule_file"], epoch_tmp)
                sess.run(tf.assign(init_lr, lr))
                sess.run(tf.assign(num_epoch, epoch_tmp))
                lr_tmp = lr
                if learning_rate is not None:
                    lr_tmp = sess.run(learning_rate)
                sess.run(tf.assign(run_lr, lr_tmp))
                run_lr_all.append(lr_tmp)

            start_time = time.time()
            feed_dict = {inputs: img_data, labels: y_true, phase_train_placeholder: True}

            loss_, _ = sess.run([yolo_loss, train_op], feed_dict=feed_dict)

            avg_loss = avg_loss * 0.9 + 0.1 * loss_
            if batch_number % 100 == 0 or batch_number == (epoch_size - 1):
                duration = time.time() - start_time
                logging.info("Epoch: [%d][%d/%d] 时间 %.3f秒 Loss %.3f 学习率 %.5f" %
                             (epoch_tmp, batch_number + 1, epoch_size, duration, avg_loss, sess.run(run_lr)))

            # 保存训练 loss
            if batch_number == 0:
                with open(loss_txt, "a") as f_loss:
                    f_loss.write("%f\n" % avg_loss)

            batch_number = (batch_number + 1) % epoch_size

            # 保存模型
            if epoch_size == 1 and (epoch >= max_epochs - 2 or os.path.exists(stop_path) or os.path.exists(delete_path)):
                save_variables_and_metagraph(sess, saver, summary_writer, log_dir, "det", epoch)
            elif batch_number == 1 and (epoch_tmp % 40 == 0 or epoch >= max_epochs - 2 or os.path.exists(stop_path) or os.path.exists(delete_path)):
                save_variables_and_metagraph(sess, saver, summary_writer, log_dir, "det", epoch)

            if batch_number == 1 and (os.path.exists(stop_path) or os.path.exists(delete_path)):
                logging.info("检测到停止文件, 停止训练")
                return log_dir

    return log_dir


def parse_arguments(argv):
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file_path", type=str, help="json 配置文件路径", default="cfg/widerface_cfg.json")
    parser.add_argument("--gpu_memory_fraction", type=float, help="GPU 内存占用上限", default=1.0)
    parser.add_argument("--gpus", type=str, help="使用的 GPU ID", default="0")
    parser.add_argument("--depth_multiplier", type=float, help="网络深度倍增系数", default=1.0)
    parser.add_argument("--optimizer", type=str, help="优化器类型", default="Adam")
    parser.add_argument("--bs", type=int, help="batch 大小", default=32)
    parser.add_argument("--lr_policy", type=str, help="学习率策略", default="poly")
    parser.add_argument("--pretrained_model", type=str, help="预训练模型路径")
    parser.add_argument("--pretrain_with_det", type=ast.literal_eval, help="是否保留检测层参数", default=False)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    params = base_func.read_hyperparams_json(args.cfg_file_path)
    logging.info("GPU 设备 ID: %s" % args.gpus)
    main(args, params)

"""训练配置 JSON 文件生成模块。

根据命令行参数（类别数、轮数、batch size、深度乘子等），
从模板配置文件生成实际训练用的 cfg JSON 文件，
同时生成学习率调度文件 lr.txt。
"""

import argparse
import os

# 原始模板配置文件路径
ori_train_config_path = "../cfg/cfg.json"
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Rewrite train.config for specific usage.")
    parser.add_argument(
        "-o",
        "--train_config_out_dir",
        default=r"../txt/",
        type=str,
        help="folder dir to new cfg.json, must be same with labels",
    )
    parser.add_argument("-c", "--class_number", default=3, type=int, help="training class number")
    parser.add_argument("-m", "--max_epoches", default=300, type=int, help="training epoch number")
    parser.add_argument("-bs", "--batch_size", default=8, type=int, help="training_batch_size")
    parser.add_argument("-d", "--depth_multiplier", default=0.5, type=float, help="0.125/0.25/0.5")
    parser.add_argument("-ckpt", "--ckpt_save_dir", default=r"../ckpt", type=str, help="folder dir to training ckpt")
    args = parser.parse_args()

    # 读取模板配置文件
    with open(ori_train_config_path, "r") as f:
        ori_config = f.readlines()
    new_config = ori_config.copy()
    train_config_out_dir = args.train_config_out_dir
    class_number = args.class_number
    batch_size = args.batch_size
    max_epoches = args.max_epoches
    # 生成学习率调度文件
    train_lr_txt = os.path.join(train_config_out_dir, "lr.txt")
    with open(train_lr_txt, "w") as f:
        f.write("0:  0.001\n")                            # 初始学习率
        f.write("%d: 0.0001\n" % (max_epoches // 2))      # 中期降低学习率
        f.write("%d:  -1" % (max_epoches + 1))             # 训练结束标记
    # 根据深度乘子生成配置文件名
    new_train_config_name = "cfg_%d.json" % (1000 * args.depth_multiplier)
    new_train_config_path = os.path.join(train_config_out_dir, new_train_config_name)
    # 替换配置模板中的占位符为实际值
    train_config_out_dir = train_config_out_dir.replace("\\", "/")
    new_config[1] = new_config[1].replace("path_to_file", train_config_out_dir)     # 训练集路径
    new_config[2] = new_config[2].replace("path_to_file", train_config_out_dir)     # 验证集路径
    # 替换 batch_size
    batch_size_str = new_config[5].split(":")[-1].strip()
    new_config[5] = new_config[5].replace(batch_size_str, str(batch_size) + ",")
    new_config[6] = new_config[6].replace("../training_policy/lr", train_config_out_dir)  # 学习率文件路径
    # 替换 max_epoches
    max_epoches_str = new_config[7].split(":")[-1].strip()
    new_config[7] = new_config[7].replace(max_epoches_str, str(max_epoches) + ",")
    # 替换类别数
    class_num_str = new_config[19].split(":")[-1].strip()
    new_config[19] = new_config[19].replace(class_num_str, str(class_number) + ",")
    # 设置检查点保存目录
    ckpt_save_dir = args.ckpt_save_dir
    if not os.path.exists(ckpt_save_dir):
        os.mkdir(ckpt_save_dir)
    ckpt_save_str = os.path.join(ckpt_save_dir, "ckpt_save_%d" % (1000 * args.depth_multiplier))
    ckpt_save_str = ckpt_save_str.replace("\\", "/")
    new_config[24] = new_config[24].replace("ckpt_save", ckpt_save_str)
    # 替换 anchor 参数
    anchor_path = os.path.join(train_config_out_dir, "anchor.txt")
    with open(anchor_path, "r") as f:
        anchors = f.readlines()
    new_config[20] = new_config[20].replace('"anchors_to_replace"', anchors[0].strip())
    # 写入新的配置文件
    with open(new_train_config_path, "w") as f:
        f.writelines(new_config)

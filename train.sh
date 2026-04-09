#!/bin/bash

# ============================================
# K210 目标检测训练脚本
# 用法: ./train.sh <dataset_name> [steps...]
# 示例:
#   ./train.sh three_fruit all          # 执行所有步骤
#   ./train.sh three_fruit 1 2 3        # 执行指定步骤
# ============================================

# 获取脚本所在目录作为工作区根目录
WORKSPACE_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 解析参数
if [ $# -lt 1 ]; then
    echo "用法: ./train.sh <dataset_name> [steps...]"
    echo "示例:"
    echo "  ./train.sh three_fruit all          # 执行所有步骤"
    echo "  ./train.sh three_fruit 1 2 3        # 执行指定步骤"
    echo ""
    echo "步骤说明:"
    echo "  1 = xml2txt (XML标注转TXT格式)"
    echo "  2 = shuffle_txt (打乱数据集)"
    echo "  3 = generate_anchors (生成锚框)"
    echo "  4 = rewrite_config (重写配置文件)"
    echo "  5 = training (模型训练)"
    echo "  6 = ckpt2pb2tflite2kmodel (模型转换)"
    exit 1
fi

DATASET_NAME="$1"
shift
STEPS=("$@")

# 默认执行所有步骤
if [ ${#STEPS[@]} -eq 0 ]; then
    STEPS=("all")
fi

# 路径配置 (相对于工作区根目录)
DATASET_DIR="${WORKSPACE_ROOT}/dataset/${DATASET_NAME}"
DATASET_PROCESS_DIR="${WORKSPACE_ROOT}/dataset_process/${DATASET_NAME}"
OUTPUT_DIR="${WORKSPACE_ROOT}/output"
CODE_DIR="${WORKSPACE_ROOT}/k210_training_code"

# 验证数据集是否存在
if [ ! -d "${DATASET_DIR}" ]; then
    echo "错误: 数据集不存在: ${DATASET_DIR}"
    exit 1
fi

echo "=========================================="
echo "工作区根目录: ${WORKSPACE_ROOT}"
echo "数据集名称: ${DATASET_NAME}"
echo "数据集路径: ${DATASET_DIR}"
echo "处理数据路径: ${DATASET_PROCESS_DIR}"
echo "执行步骤: ${STEPS[@]}"
echo "=========================================="

# 创建必要的输出目录
mkdir -p "${DATASET_PROCESS_DIR}"
mkdir -p "${OUTPUT_DIR}/txt/detection/${DATASET_NAME}"
mkdir -p "${OUTPUT_DIR}/ckpt/detection/${DATASET_NAME}"
mkdir -p "${OUTPUT_DIR}/results/detection/${DATASET_NAME}"

# 判断是否执行某一步
should_run() {
    if [[ " ${STEPS[@]} " =~ " $1 " || " ${STEPS[@]} " =~ " all " ]]; then
        return 0
    else
        return 1
    fi
}

# ============================================
# Step 1: XML标注转TXT格式
# ============================================
if should_run 1; then
    echo ""
    echo "[Step 1/6] XML标注转TXT格式"
    echo "-------------------------------------------"
    
    cd "${CODE_DIR}/detection/utils/scripts" || exit
    
    python3 -u xml2txt.py \
        -i "${DATASET_DIR}/images" \
        -a "${DATASET_DIR}/xml" \
        -sbd "${OUTPUT_DIR}/txt/detection/${DATASET_NAME}" \
        -sid "${DATASET_PROCESS_DIR}/images_jpg" \
        -sxd "${DATASET_PROCESS_DIR}/xml_jpg"
    
    echo "[Step 1] 完成"
fi

# ============================================
# Step 2: 打乱TXT数据集
# ============================================
if should_run 2; then
    echo ""
    echo "[Step 2/6] 打乱TXT数据集"
    echo "-------------------------------------------"
    
    cd "${CODE_DIR}/detection/utils/scripts" || exit
    
    python3 -u shuffle_txt.py \
        -i "${OUTPUT_DIR}/txt/detection/${DATASET_NAME}" \
        -a "${DATASET_PROCESS_DIR}/xml_jpg" \
        -t "${DATASET_PROCESS_DIR}/images_jpg" \
        -s "${DATASET_PROCESS_DIR}/validation"
    
    echo "[Step 2] 完成"
fi

# ============================================
# Step 3: 生成YOLO锚框
# ============================================
if should_run 3; then
    echo ""
    echo "[Step 3/6] 生成YOLO锚框"
    echo "-------------------------------------------"
    
    cd "${CODE_DIR}/detection/utils/scripts" || exit
    
    python3 -u gen_yolov3_anchor.py \
        -i "${OUTPUT_DIR}/txt/detection/${DATASET_NAME}" \
        -n 10 \
        -tw 320 \
        -th 240
    
    echo "[Step 3] 完成"
fi

# ============================================
# Step 4: 重写配置文件JSON
# ============================================
if should_run 4; then
    echo ""
    echo "[Step 4/6] 重写配置文件JSON"
    echo "-------------------------------------------"
    
    # 从labels.txt读取类别数
    LABELS_FILE="${DATASET_DIR}/labels.txt"
    if [ ! -f "${LABELS_FILE}" ]; then
        echo "[错误] 找不到labels.txt文件: ${LABELS_FILE}"
        exit 1
    fi
    
    # 计算类别数（统计非空行数）
    NUM_CLASSES=$(grep -cve '^\s*$' "${LABELS_FILE}")
    echo "[INFO] 从labels.txt读取到类别数: ${NUM_CLASSES}"
    echo "[INFO] 类别列表:"
    while IFS= read -r line; do
        # 跳过空行
        if [ -n "$(echo "$line" | tr -d '[:space:]')" ]; then
            echo "  - $line"
        fi
    done < "${LABELS_FILE}"
    
    cd "${CODE_DIR}/detection/utils/scripts" || exit
    
    python3 -u rewrite_cfg_json.py \
        -o "${OUTPUT_DIR}/txt/detection/${DATASET_NAME}" \
        -m 100 \
        -c "${NUM_CLASSES}" \
        -d 0.25 \
        -ckpt "${OUTPUT_DIR}/ckpt/detection/${DATASET_NAME}"
    
    echo "[Step 4] 完成"
fi

# ============================================
# Step 5: 模型训练
# ============================================
if should_run 5; then
    echo ""
    echo "[Step 5/6] 模型训练"
    echo "-------------------------------------------"
    
    cd "${CODE_DIR}/detection/code" || exit
    
    python3 -u train_yolo_lite.py \
        --gpus 0 \
        --optimizer adam \
        "${OUTPUT_DIR}/txt/detection/${DATASET_NAME}/cfg_250.json" \
        --depth_multiplier 0.25 \
        --pretrained_model "${CODE_DIR}/detection/ckpt_restore/MobileNet_v1_0.25" \
        --bs 8 \
        --lr_policy step
    
    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        echo "[Step 5] 完成"
    else
        echo "[Step 5] 训练失败！请检查上述错误信息"
        echo "提示: 如果Step 5失败，Step 6将无法执行"
        exit 1
    fi
fi

# ============================================
# Step 6: 模型转换 (ckpt -> pb -> tflite -> kmodel)
# ============================================
if should_run 6; then
    echo ""
    echo "[Step 6/6] 模型转换 (ckpt -> pb -> tflite -> kmodel)"
    echo "-------------------------------------------"
    
    # 检查训练输出是否存在
    CKPT_DIR="${OUTPUT_DIR}/ckpt/detection/${DATASET_NAME}/ckpt_save_250"
    if [ ! -d "${CKPT_DIR}" ]; then
        echo "[错误] 训练输出目录不存在: ${CKPT_DIR}"
        echo "[错误] 请先成功完成Step 5训练步骤"
        exit 1
    fi
    
    # 检查是否有checkpoint文件
    if ! ls "${CKPT_DIR}"/*.meta 1> /dev/null 2>&1; then
        echo "[错误] 未找到checkpoint文件 (*.meta)"
        echo "[错误] Step 5训练可能未成功完成，请先检查训练日志"
        exit 1
    fi
    
    cd "${CODE_DIR}/detection/code" || exit
    
    python3 -u pb2tflite2kmodel.py \
        det_ckpt_save_name \
        "${OUTPUT_DIR}/txt/detection/${DATASET_NAME}/cfg_250.json" \
        "${OUTPUT_DIR}/results/detection/${DATASET_NAME}" \
        "${DATASET_PROCESS_DIR}/validation/val_images" \
        -m 300
    
    if [ $? -eq 0 ]; then
        echo "[Step 6] 完成"
    else
        echo "[Step 6] 模型转换失败！请检查上述错误信息"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "所有指定步骤执行完毕!"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

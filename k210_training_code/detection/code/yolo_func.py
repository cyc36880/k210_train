"""YOLO 核心功能模块。

提供 YOLO 目标检测所需的核心函数，包括：
- 数据增强与预处理（缩放、平移、颜色扰动）
- YOLO 检测头（特征图转换为边界框参数）
- 边界框校正与 IoU 计算
- 真实框预处理（将标注转换为训练格式）
- YOLO 损失函数（包含 xy、wh、置信度、分类损失）
- 数据生成器（批量生成训练数据）
- 类别和 anchor 加载工具
"""

import numpy as np
import tensorflow as tf
from keras import backend as K
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image


def compose(*funcs):
    """将多个函数从左到右依次组合。

    参考: https://mathieularose.com/function-composition-in-python/

    参数:
        *funcs: 需要组合的函数序列

    返回:
        组合后的函数

    异常:
        ValueError: 当函数序列为空时抛出
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


def letterbox_image(image, size):
    """保持宽高比缩放图像并用灰色填充。

    将图像等比例缩放到目标尺寸，不足部分用灰色 (128,128,128) 填充，
    确保图像不变形。

    参数:
        image: PIL.Image 对象，输入图像
        size: tuple (w, h)，目标尺寸

    返回:
        PIL.Image: 缩放并填充后的图像
    """
    iw, ih = image.size
    w, h = size
    scale = min(w / float(iw), h / float(ih))
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def rand(a=0, b=1):
    """生成 [a, b) 范围内的随机浮点数。

    参数:
        a: 下界，默认 0
        b: 上界，默认 1

    返回:
        float: [a, b) 范围内的随机数
    """
    return np.random.rand() * (b - a) + a


def get_random_data(
    annotation_line, input_shape, random=True, max_boxes=20, jitter=0.3, hue=0.1, sat=1.5, val=1.5, proc_img=True
):
    """实时数据增强的随机预处理。

    对输入图像进行随机缩放、平移和颜色扰动（HSV 空间），
    同时对应调整边界框坐标。

    参数:
        annotation_line: str，标注行，格式为 '图片路径 x1,y1,x2,y2,cls ...'
        input_shape: tuple (h, w)，网络输入尺寸
        random: bool，是否启用随机增强
        max_boxes: int，单张图像最大目标数
        jitter: float，抖动比例（未使用）
        hue: float，色调扰动范围
        sat: float，饱和度扰动范围
        val: float，明度扰动范围
        proc_img: bool，是否处理图像（未使用）

    返回:
        image_data: ndarray，增强后的图像数据，值域 [0, 1]
        box_data: ndarray，shape=(max_boxes, 5)，调整后的边界框 [x1,y1,x2,y2,cls]
    """
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    boxes = np.array([np.array(list(map(float, box.split(",")))) for box in line[1:]])

    # 随机缩放图像
    scale = np.random.uniform() / 10.0 + 1.0
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.NEAREST)

    # 随机平移图像
    max_offx = (scale - 1.0) * iw
    max_offy = (scale - 1.0) * ih
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    image = image.crop([offx, offy, (offx + iw), (offy + ih)])
    image = image.resize((w, h), Image.NEAREST)

    # 颜色扰动（在 HSV 空间进行色调、饱和度、明度的随机调整）
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < 0.5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < 0.5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.0)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # 将 HSV 转回 RGB，值域 [0, 1]

    # 根据缩放和平移参数调整边界框坐标
    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2, class_id = box
        x1 = int((x1 * scale - offx) / iw * w)
        x2 = int((x2 * scale - offx) / iw * w)

        y1 = int((y1 * scale - offy) / ih * h)
        y2 = int((y2 * scale - offy) / ih * h)

        x1 = max(min(x1, w - 1), 0)
        x2 = max(min(x2, w - 1), 0)
        y1 = max(min(y1, h - 1), 0)
        y2 = max(min(y2, h - 1), 0)

        new_boxes.append([x1, y1, x2, y2, class_id])
    box = np.array(new_boxes)
    # 过滤无效框并填充到固定大小数组
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # 丢弃宽高小于1的无效框
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[: len(box)] = box
    return image_data, box_data


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """将网络最终层特征转换为边界框参数。

    将卷积特征图解码为边界框的中心坐标、宽高、置信度和类别概率。

    参数:
        feats: Tensor，网络输出特征图
        anchors: list，anchor 尺寸列表 [[w, h], ...]
        num_classes: int，类别数量
        input_shape: Tensor，网络输入尺寸 [h, w]
        calc_loss: bool，是否用于损失计算（返回额外的中间结果）

    返回:
        若 calc_loss=True: (grid, feats, box_xy, box_wh)
        若 calc_loss=False: (box_xy, box_wh, box_confidence, box_class_probs)
    """
    num_anchors = len(anchors)
    # 将 anchor 重塑为 [1, 1, 1, num_anchors, 2] 用于广播计算
    anchors_tensor = K.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # 获取特征图的高和宽
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 将预测值调整到对应的网格位置和 anchor 尺寸
    # box_xy: 中心坐标相对于整张图的比例
    box_xy = (K.cast(K.sigmoid(feats[..., :2]), K.dtype(feats)) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """将预测框坐标校正到原始图像尺寸。

    考虑 letterbox 缩放引入的偏移和缩放，将网络输出的
    归一化坐标转换回原始图像的像素坐标。

    参数:
        box_xy: Tensor，边界框中心坐标（归一化）
        box_wh: Tensor，边界框宽高（归一化）
        input_shape: Tensor，网络输入尺寸 [h, w]
        image_shape: Tensor，原始图像尺寸 [h, w]

    返回:
        boxes: Tensor，校正后的边界框 [y_min, x_min, y_max, x_max]（像素坐标）
    """
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2.0 / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    # box_mins = box_yx - (box_hw / 2.)
    # box_maxes = box_yx + (box_hw / 2.)
    box_mins = box_yx * image_shape - (box_hw / 2.0) * image_shape + 1
    box_mins = tf.maximum(box_mins, 1)
    box_maxes = box_yx * image_shape + (box_hw / 2.0) * image_shape + 1
    # box_maxes = tf.minimum(box_maxes,image_shape)
    boxes = K.concatenate(
        [
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            tf.minimum(box_maxes[..., 0:1], image_shape[0]),  # y_max
            tf.minimum(box_maxes[..., 1:2], image_shape[1]),  # x_max
        ]
    )

    # 返回原始图像尺寸下的像素坐标
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """处理卷积层输出，提取边界框和分数。

    将网络特征图解码为校正后的边界框坐标和对应的类别分数。

    参数:
        feats: Tensor，网络输出特征图
        anchors: list，anchor 尺寸列表
        num_classes: int，类别数量
        input_shape: Tensor，网络输入尺寸
        image_shape: Tensor，原始图像尺寸

    返回:
        boxes: Tensor，shape=(-1, 4)，校正后的边界框
        box_scores: Tensor，shape=(-1, num_classes)，各类别分数
    """
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # print(box_xy)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def preprocess_true_boxes(true_boxes, input_shape,grid_shapes, anchors, num_classes):
    """将真实标注框预处理为训练所需的格式。

    将原始的 [x_min, y_min, x_max, y_max, class_id] 格式转换为
    YOLO 训练所需的网格化标签，包括中心坐标、宽高、置信度和类别独热编码。

    参数:
        true_boxes: ndarray, shape=(m, T, 5)，真实框坐标（像素值）
                    每个框格式为 [x_min, y_min, x_max, y_max, class_id]
        input_shape: array-like, [h, w]，网络输入尺寸，需为 32 的倍数
        grid_shapes: list, [grid_h, grid_w]，特征图网格尺寸
        anchors: ndarray, shape=(N, 2)，anchor 宽高
        num_classes: int，类别数量

    返回:
        y_true: ndarray, shape=(m, grid_h, grid_w, num_anchors, 5+num_classes)
                xywh 为相对值，第4维为置信度，之后为类别独热编码
    """
    # 验证类别 ID 合法性
    assert (true_boxes[..., 4] < num_classes).all(), "class id must be less than num_classes"
    num_anchors = len(anchors)

    true_boxes = np.array(true_boxes, dtype="float32")
    input_shape = np.array(input_shape, dtype="int32")
    # 将绝对坐标转换为相对坐标（中心点 xy 和宽高 wh）
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2.0
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    # grid_shapes = input_shape//32
    grid_shapes = [grid_shapes[0], grid_shapes[1]]
    y_true = np.zeros((m, grid_shapes[0], grid_shapes[1], num_anchors, 5 + num_classes), dtype="float32")

    # 扩展维度以便广播计算 IoU
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.0
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0  # 过滤宽度为0的无效框

    for b in range(m):
        # 丢弃宽高为零的无效行
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # 扩展维度用于广播计算 IoU
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.0
        box_mins = -box_maxes

        # 计算每个真实框与所有 anchor 的 IoU
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # 为每个真实框找到 IoU 最大的 anchor
        best_anchor = np.argmax(iou, axis=-1)

        # 将每个真实框分配到对应的网格和 anchor 位置
        for t, n in enumerate(best_anchor):
            i = np.floor(true_boxes[b, t, 0] * grid_shapes[1]).astype("int32")  # 网格 x 索引
            j = np.floor(true_boxes[b, t, 1] * grid_shapes[0]).astype("int32")  # 网格 y 索引
            c = true_boxes[b, t, 4].astype("int32")  # 类别 ID
            y_true[b, j, i, n, 0:4] = true_boxes[b, t, 0:4]  # 填入 xywh
            y_true[b, j, i, n, 4] = 1  # 置信度标记为1
            y_true[b, j, i, n, 5 + c] = 1  # 对应类别设为1（独热编码）

    return y_true


def box_iou(b1, b2):
    """计算两组边界框之间的 IoU（交并比）。

    参数:
        b1: Tensor, shape=(i1,...,iN, 4)，格式为 [x, y, w, h]
        b2: Tensor, shape=(j, 4)，格式为 [x, y, w, h]

    返回:
        iou: Tensor, shape=(i1,...,iN, j)，每对框之间的 IoU 值
    """

    # 扩展 b1 维度用于广播，计算 b1 的左上角和右下角坐标
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.0
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 扩展 b2 维度用于广播，计算 b2 的左上角和右下角坐标
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.0
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def yolo_loss(input_shape,grid_shape, yolo_outputs, y_true, anchors, num_classes, ignore_thresh=0.5, print_loss=False):
    """计算 YOLO 检测损失。

    包含四个损失分量：
    - xy_loss: 中心坐标损失
    - wh_loss: 宽高损失
    - confidence_loss: 置信度损失（含正负样本）
    - class_loss: 分类损失

    参数:
        input_shape: tuple，网络输入尺寸 (h, w)
        grid_shape: tuple，特征图网格尺寸 (grid_h, grid_w)
        yolo_outputs: Tensor，网络输出特征图
        y_true: Tensor，预处理后的真实标签
        anchors: ndarray, shape=(N, 2)，anchor 宽高
        num_classes: int，类别数量
        ignore_thresh: float，IoU 忽略阈值，低于此值的负样本参与置信度损失
        print_loss: bool，是否打印各损失分量

    返回:
        loss: Tensor, shape=(1,)，总损失值
    """

    # input_shape = K.cast(K.shape(yolo_outputs)[1:3] * 32, K.dtype(y_true[0]))
    input_shape = K.cast([input_shape[0], input_shape[1]], K.dtype(y_true[0]))
    # grid_shapes = K.cast(K.shape(yolo_outputs)[1:3], K.dtype(y_true[0]))
    grid_shapes = K.cast([grid_shape[0], grid_shape[1]], K.dtype(y_true[0]))
    # print(grid_shapes)
    m = K.shape(yolo_outputs)[0]  # 批量大小
    mf = K.cast(m, K.dtype(yolo_outputs))

    object_mask = y_true[..., 4:5]  # 目标存在掩码

    # 通过 yolo_head 解码网络输出
    grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs, anchors, num_classes, input_shape, calc_loss=True)
    pred_box = K.concatenate([pred_xy, pred_wh])

    # 对原始预测值应用激活函数
    pred_xy = K.sigmoid(raw_pred[..., :2])           # 中心坐标 sigmoid
    pred_confidence = K.sigmoid(raw_pred[..., 4:5])   # 置信度 sigmoid
    pred_class_probs = K.softmax(raw_pred[..., 5:])   # 类别概率 softmax

    # 将真实框转换为 Darknet 原始格式用于计算损失
    raw_true_xy = y_true[..., :2] * grid_shapes[::-1] - grid
    raw_true_wh = K.log(y_true[..., 2:4] / anchors * input_shape[::-1])
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # 避免 log(0)=-inf
    box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]  # 大框权重小，小框权重大

    # 构建忽略掩码：IoU 超过阈值的负样本不参与置信度损失
    ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
    object_mask_bool = K.cast(object_mask, "bool")

    def loop_body(b, ignore_mask):
        true_box = tf.boolean_mask(y_true[b, ..., 0:4], object_mask_bool[b, ..., 0])
        iou = box_iou(pred_box[b], true_box)
        best_iou = K.max(iou, axis=-1)
        ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
        return b + 1, ignore_mask

    _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = K.expand_dims(ignore_mask, -1)
    # 计算各损失分量的残差
    delta_xy = object_mask * box_loss_scale * (raw_true_xy - pred_xy)       # 中心坐标残差
    delta_wh = object_mask * box_loss_scale * (raw_true_wh - raw_pred[..., 2:4])  # 宽高残差
    delta_con = object_mask * (1 - pred_confidence) + (1 - object_mask) * (0 - pred_confidence) * ignore_mask  # 置信度残差
    delta_class = object_mask * (y_true[..., 5:] - pred_class_probs)        # 分类残差

    delta = tf.concat([delta_xy, delta_wh, delta_con, delta_class], axis=-1)

    loss = tf.reduce_sum(tf.square(delta)) / mf
    xy_loss = tf.reduce_sum(tf.square(delta_xy)) / mf
    wh_loss = tf.reduce_sum(tf.square(delta_wh)) / mf
    confidence_loss = tf.reduce_sum(tf.square(delta_con)) / mf
    class_loss = tf.reduce_sum(tf.square(delta_class)) / mf

    if print_loss:
        loss = tf.Print(
            loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message="loss: "
        )

    return loss


def data_generator(annotation_lines, batch_size, epoch_num, input_shape,grid_shape, anchors, num_classes):
    """训练数据生成器。

    按批次生成训练数据，每个 epoch 开始时打乱样本顺序。

    参数:
        annotation_lines: list，标注行列表
        batch_size: int，批量大小
        epoch_num: int，总训练轮数
        input_shape: tuple (h, w)，网络输入尺寸
        grid_shape: tuple (grid_h, grid_w)，特征图网格尺寸
        anchors: ndarray，anchor 尺寸
        num_classes: int，类别数量

    生成:
        [image_data, y_true]: 图像数据和对应的训练标签
        epoch: 当前轮次
    """
    n = len(annotation_lines)
    i = 0
    epoch = 0
    while epoch < epoch_num:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            if i == n - 1:
                epoch += 1
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            # image, box = get_random_data(annotation_lines[i], input_shape, random=False)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape,grid_shape, anchors, num_classes)
        yield [image_data, y_true], epoch


def data_generator_wrapper(annotation_lines, batch_size, epoch_num, input_shape, grid_shape,anchors, num_classes):
    """数据生成器包装函数。

    对输入参数进行有效性检查后返回数据生成器。

    参数:
        annotation_lines: list，标注行列表
        batch_size: int，批量大小
        epoch_num: int，总训练轮数
        input_shape: tuple，网络输入尺寸
        grid_shape: tuple，特征图网格尺寸
        anchors: ndarray，anchor 尺寸
        num_classes: int，类别数量

    返回:
        data_generator 或 None（当数据为空或 batch_size 无效时）
    """
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, epoch_num, input_shape, grid_shape,anchors, num_classes)


def get_classes(classes_path):
    """从文件加载类别名称列表。

    参数:
        classes_path: str，类别文件路径，每行一个类别名

    返回:
        list: 类别名称列表
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """从文件加载 anchor 尺寸。

    参数:
        anchors_path: str，anchor 文件路径，逗号分隔的浮点数

    返回:
        ndarray: shape=(-1, 2)，每行为一个 anchor 的 [宽, 高]
    """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(",")]
    return np.array(anchors).reshape(-1, 2)

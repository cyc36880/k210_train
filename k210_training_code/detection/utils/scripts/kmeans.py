"""基于 IoU 距离的 K-Means 聚类模块。

用于生成 YOLO 目标检测模型的 anchor 框，
使用 IoU 作为距离度量而非欧氏距离，更适合目标检测场景。
"""

import numpy as np


def iou(box, clusters):
    """计算一个框与 k 个聚类中心之间的 IoU。

    注意：框和聚类中心都已平移到原点，只使用宽和高。

    参数:
        box: tuple 或 array，单个框的 [宽, 高]
        clusters: ndarray, shape=(k, 2)，k 个聚类中心的 [宽, 高]

    返回:
        ndarray, shape=(k,)，与每个聚类中心的 IoU 值

    异常:
        ValueError: 当框面积为零时抛出
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    # 计算交集面积
    intersection = x * y
    # 计算并集面积
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """计算一组框与 k 个聚类中心之间的平均 IoU。

    每个框取与所有聚类中心中最大的 IoU，然后求均值。

    参数:
        boxes: ndarray, shape=(r, 2)，r 个框的 [宽, 高]
        clusters: ndarray, shape=(k, 2)，k 个聚类中心

    返回:
        float: 平均 IoU 值
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """将所有框平移到原点。

    将 [x1, y1, x2, y2] 格式转换为 [w, h] 格式。

    参数:
        boxes: ndarray, shape=(r, 4)，[x1, y1, x2, y2] 格式

    返回:
        ndarray, shape=(r, 2)，[宽, 高] 格式
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """使用 IoU 作为距离度量的 K-Means 聚类。

    通过迭代更新聚类中心，找到最优的 k 个 anchor 尺寸。

    参数:
        boxes: ndarray, shape=(r, 2)，r 个框的 [宽, 高]
        k: int，聚类数量
        dist: callable，聚类中心更新函数，默认为 np.median（中位数）

    返回:
        ndarray, shape=(k, 2)，k 个聚类中心的 [宽, 高]
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))    # 距离矩阵
    last_clusters = np.zeros((rows,))  # 上一次的聚类分配结果

    np.random.seed()

    # Forgy 初始化：随机选择 k 个样本作为初始聚类中心
    clusters = boxes[np.random.choice(rows, k, replace=(k >= rows))]

    while True:
        # 计算每个框到每个聚类中心的距离（1 - IoU）
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        # 找到每个框最近的聚类中心
        nearest_clusters = np.argmin(distances, axis=1)

        # 如果聚类分配不再变化，则收敛退出
        if (last_clusters == nearest_clusters).all():
            break

        # 更新聚类中心（使用中位数或其他统计量）
        for cluster in range(k):
            if np.sum(nearest_clusters == cluster) == 0:
                continue
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

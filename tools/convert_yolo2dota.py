# Copyright (c) OpenMMLab. All rights reserved.
"""YOLO -> DOTA v1 标注转换脚本（只读原始数据集，输出到新目录，不修改原始数据）。

原始数据（比赛提供）:
    data/训练集/{visible,infrared,depth}/<img_id>.{jpg,png}
    data/训练集/labels/<img_id>.txt    # YOLO 格式: class cx cy w h (归一化)
    data/测试集/{visible,infrared,depth}/<img_id>.{jpg,png}   # 无 labels

转换输出（DOTA v1 格式）:
    data_dota/训练集/labels/<img_id>.txt   # x1 y1 x2 y2 x3 y3 x4 y4 class 0

用法:
    python tools/convert_yolo2dota.py [--data-root DATA_ROOT] [--out-root OUT_ROOT]

说明:
    1. 反归一化需要图片真实宽高，转换时逐张读取 visible 图像尺寸。
    2. 类别名不允许含空格（DOTA 解析按空白 split），garbage can -> garbage_can。
    3. 原标注为水平框，转换后 4 角点 = (x1,y1),(x2,y1),(x2,y2),(x1,y2)，即 angle=0 的旋转框。
"""
import argparse
import glob
import os
import os.path as osp

import cv2
import numpy as np

# ========== 脚本内固化配置 ==========
DATA_ROOT = r"D:\我太想进步了\竞赛\人工智能算法挑战赛\data"
OUT_ROOT = r"D:\我太想进步了\竞赛\人工智能算法挑战赛\data_dota"
SPLITS = ("训练集",)  # 训练集才需要转换 labels；测试集无标注
LABELS_DIR_NAME = "labels"
VISIBLE_DIR_NAME = "visible"

# 12 类映射（比赛类别定义，0-11）
CLS_MAP = {
    0: "person",
    1: "boat",
    2: "animal",
    3: "seat",
    4: "sign",
    5: "bicycle",
    6: "car",
    7: "ball",
    8: "light",
    9: "garbage_can",  # 原 'garbage can'，空格改为下划线（DOTA 解析按空白分列）
    10: "uav",
    11: "tricycle",
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def get_img_size(img_dir, img_id):
    """按 id 查找图片并返回 (w, h)。用 imdecode 兼容中文路径。"""
    for ext in IMG_EXTS:
        p = osp.join(img_dir, img_id + ext)
        if osp.exists(p):
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                return w, h
            raise RuntimeError(f"图片无法读取: {p}")
    raise FileNotFoundError(f"未找到图片 {img_id} 于 {img_dir}")


def yolo_to_dota(yolo_path, img_dir):
    """把单张 YOLO 标注转成 DOTA 行列表。"""
    img_id = osp.splitext(osp.basename(yolo_path))[0]
    w, h = get_img_size(img_dir, img_id)
    dota_lines = []
    with open(yolo_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"[warn] 跳过非法行 {yolo_path}: {line!r}")
                continue
            cls_id, cx, cy, bw, bh = (float(parts[0]), float(parts[1]),
                                      float(parts[2]), float(parts[3]),
                                      float(parts[4]))
            cls_name = CLS_MAP.get(int(cls_id))
            if cls_name is None:
                print(f"[warn] 未知类别 {int(cls_id)} in {yolo_path}，跳过")
                continue
            # 归一化 -> 像素坐标（水平框 4 角点，顺时针）
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            dota_lines.append(
                " ".join(f"{p:.2f}" for xy in pts for p in xy) +
                f" {cls_name} 0")
    return dota_lines


def convert_split(split):
    src_labels = osp.join(DATA_ROOT, split, LABELS_DIR_NAME)
    src_visible = osp.join(DATA_ROOT, split, VISIBLE_DIR_NAME)
    dst_labels = osp.join(OUT_ROOT, split, LABELS_DIR_NAME)
    os.makedirs(dst_labels, exist_ok=True)

    yolo_files = sorted(glob.glob(osp.join(src_labels, "*.txt")))
    print(f"[{split}] 共 {len(yolo_files)} 个标注文件")
    n_bbox, n_empty, n_err = 0, 0, 0
    for i, yf in enumerate(yolo_files):
        try:
            lines = yolo_to_dota(yf, src_visible)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {yf}: {e}")
            n_err += 1
            continue
        img_id = osp.splitext(osp.basename(yf))[0]
        with open(osp.join(dst_labels, img_id + ".txt"), "w") as out:
            if lines:
                out.write("\n".join(lines) + "\n")
            else:
                out.write("")  # 保留空标注（与原始空文件一致）
                n_empty += 1
        n_bbox += len(lines)
        if (i + 1) % 500 == 0:
            print(f"  ... 已转换 {i + 1}/{len(yolo_files)}")
    print(f"[{split}] 完成: 目标框 {n_bbox}，空标注 {n_empty}，失败 {n_err}")
    return n_bbox, n_empty, n_err


def main():
    global DATA_ROOT, OUT_ROOT
    parser = argparse.ArgumentParser(description="YOLO -> DOTA 标注转换")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--out-root", default=OUT_ROOT)
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    OUT_ROOT = args.out_root

    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"OUT_ROOT  = {OUT_ROOT}")
    print("类别映射:")
    for k, v in sorted(CLS_MAP.items()):
        print(f"  {k}: {v}")

    total_bbox = 0
    for split in SPLITS:
        b, _, _ = convert_split(split)
        total_bbox += b
    print(f"全部完成，总目标框数: {total_bbox}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""核对转换脚本与官方标签格式说明是否一致。

官方说明: [class_id, norm_center_x, norm_center_y, norm_w, norm_h]
  - norm_center_x/y: 中心点相对整张图宽/高的归一化位置 (0~1)
  - norm_w/h: 框宽/高占整张图像宽/高的比例 (0~1)

验证项:
  1. 全量图片尺寸分布 + 三模态(visible/infrared/depth)尺寸一致性
  2. DOTA 角点是否都在图像边界内（若越界 => 反归一化分母取错）
  3. round-trip: DOTA -> YOLO 与原始 YOLO 逐行数值对比（等价值）
"""
import glob
import os
import sys

import cv2
import numpy as np

DATA_ROOT = r"D:\我太想进步了\竞赛\人工智能算法挑战赛\data"
DOTA_ROOT = r"D:\我太想进步了\竞赛\人工智能算法挑战赛\data_dota"
SPLITS = ("训练集", "测试集")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

CLS_MAP = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign",
    5: "bicycle", 6: "car", 7: "ball", 8: "light", 9: "garbage_can",
    10: "uav", 11: "tricycle",
}
ID2NAME = {v: k for k, v in CLS_MAP.items()}


def imread_zh(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def find_img(img_dir, img_id):
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, img_id + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    # ---------- 1. 尺寸分布 & 三模态一致性 ----------
    print("=" * 60)
    print("[1] 图像尺寸分布 + 三模态一致性")
    for split in SPLITS:
        vis_dir = os.path.join(DATA_ROOT, split, "visible")
        ir_dir = os.path.join(DATA_ROOT, split, "infrared")
        dp_dir = os.path.join(DATA_ROOT, split, "depth")
        files = sorted(glob.glob(os.path.join(vis_dir, "*")))
        from collections import Counter
        size_cnt = Counter()
        mismatch = []  # (img_id, vis_size, ir_size, dp_size)
        for i, f in enumerate(files):
            img_id = os.path.splitext(os.path.basename(f))[0]
            v = imread_zh(f)
            vs = (v.shape[1], v.shape[0]) if v is not None else None
            size_cnt[vs] += 1
            if vs != (640, 360):
                # 非主流尺寸：核对其红外/深度是否同尺寸（空间对齐）
                p_ir, p_dp = find_img(ir_dir, img_id), find_img(dp_dir, img_id)
                irs = imread_zh(p_ir).shape[:2][::-1] if p_ir else None
                dps = imread_zh(p_dp).shape[:2][::-1] if p_dp else None
                if vs != irs or vs != dps:
                    mismatch.append((img_id, vs, irs, dps))
            if (i + 1) % 500 == 0:
                print(f"  {split} ... {i + 1}/{len(files)}")
        print(f"[{split}] 尺寸分布: {dict(size_cnt)}")
        print(f"[{split}] 三模态尺寸不一致: {len(mismatch)} 个")
        for m in mismatch[:10]:
            print("   ", m)

    # ---------- 2. 训练集: 角点越界检查 + round-trip ----------
    print("=" * 60)
    print("[2] 训练集 DOTA 角点越界检查 + round-trip 数值等价")
    vis_dir = os.path.join(DATA_ROOT, "训练集", "visible")
    yolo_dir = os.path.join(DATA_ROOT, "训练集", "labels")
    dota_dir = os.path.join(DOTA_ROOT, "训练集", "labels")
    total_lines = 0
    oob = 0        # 越界角点数
    rt_diff = []   # round-trip 不匹配
    bad_files = []
    yolo_files = sorted(glob.glob(os.path.join(yolo_dir, "*.txt")))
    for i, yf in enumerate(yolo_files):
        img_id = os.path.splitext(os.path.basename(yf))[0]
        p_vis = find_img(vis_dir, img_id)
        img = imread_zh(p_vis)
        w, h = img.shape[1], img.shape[0]
        dota_f = os.path.join(dota_dir, img_id + ".txt")
        with open(yf) as fh:
            yolo_lines = [l.split() for l in fh if l.strip()]
        with open(dota_f) as fh:
            dota_lines = [l.split() for l in fh if l.strip()]
        if len(yolo_lines) != len(dota_lines):
            bad_files.append((img_id, "行数不一致",
                              len(yolo_lines), len(dota_lines)))
            continue
        for yl, dl in zip(yolo_lines, dota_lines):
            total_lines += 1
            # --- 越界检查 ---
            xs = np.array([float(dl[i]) for i in (0, 2, 4, 6)])
            ys = np.array([float(dl[i]) for i in (1, 3, 5, 7)])
            if xs.min() < -1e-2 or xs.max() > w + 1e-2 \
                    or ys.min() < -1e-2 or ys.max() > h + 1e-2:
                oob += 1
            # --- 类别检查 ---
            cls_name = dl[8]
            if cls_name != CLS_MAP[int(float(yl[0]))]:
                bad_files.append((img_id, "类别不匹配", yl[0], cls_name))
                continue
            # --- round-trip: DOTA 角点 -> 归一化中心/宽高 ---
            x1, y1 = xs[0], ys[0]
            x2, y2 = xs[2], ys[2]
            r_cx, r_cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            r_w, r_h = (x2 - x1) / w, (y2 - y1) / h
            o_cx, o_cy = float(yl[1]), float(yl[2])
            o_w, o_h = float(yl[3]), float(yl[4])
            if (abs(r_cx - o_cx) > 1e-3 or abs(r_cy - o_cy) > 1e-3
                    or abs(r_w - o_w) > 1e-3 or abs(r_h - o_h) > 1e-3):
                rt_diff.append((img_id, (o_cx, o_cy, o_w, o_h),
                                (r_cx, r_cy, r_w, r_h)))
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(yolo_files)}")
    print(f"总目标框: {total_lines}")
    print(f"越界角点: {oob}")
    print(f"round-trip 不匹配: {len(rt_diff)}")
    for d in rt_diff[:5]:
        print("   ", d)
    print(f"文件级异常: {len(bad_files)}")
    for b in bad_files[:5]:
        print("   ", b)
    print("核验完成")


if __name__ == "__main__":
    main()

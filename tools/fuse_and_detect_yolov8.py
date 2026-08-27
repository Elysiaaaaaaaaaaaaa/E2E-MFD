# -*- coding: utf-8 -*-
"""
fuse_and_detect_yolov8.py
======================
用 E2E-MFD 训练好的加权融合网络（FusionNet）把测试集每一对 (visible, infrared) 融合成
一张 RGB 图像，再送入 YOLOv8 做目标检测。

依赖（已装于 venv39）：
  torch 1.13.1+cpu / numpy 1.24.4 / opencv-python 4.8.1.78 / mmcv-full 1.7.2 /
  mmdet 2.28.2 / mmrotate 0.3.4 / timm / e2cnn / ultralytics

用法：
  python tools/fuse_and_detect_yolov8.py                 # 全流程：融合 -> YOLOv8 检测
  python tools/fuse_and_detect_yolov8.py --fuse-only     # 只生成融合图
  python tools/fuse_and_detect_yolov8.py --detect-only   # 只用已生成的融合图做检测
  python tools/fuse_and_detect_yolov8.py --limit 10      # 只处理前 10 张（冒烟测试）
  python tools/fuse_and_detect_yolov8.py --mock-weights  # 不加载 E2E-MFD 权重，随机权重跑通流程
  python tools/fuse_and_detect_yolov8.py --check         # 只检查依赖/路径/权重，不做处理

说明：
  - 融合网络是 E2E-MFD 的 FusionNet（HSV 空间逐像素加权融合），前向逻辑在本脚本内独立
    实现（复用 Oriented_rcnn_m 的 backbone/neck/fusion），不依赖仓库里写死
    DroneVehicle 尺寸与 cuda 的 forward_fusion。
  - 融合图保持测试集原尺寸（全卷积网络可任意尺寸），YOLOv8 内部自己做 letterbox。
  - 测试集无标注，YOLOv8 输出 YOLO 格式检测 txt（归一化）+ 可视化图。
"""

import argparse
import os
import sys

import numpy as np
import cv2

# ----------------------------------------------------------------------------
# CONFIG（脚本内固化，按需修改）
# ----------------------------------------------------------------------------
CONFIG = dict(
    # ---- E2E-MFD 融合侧 ----
    # mmrotate 模型配置文件（Oriented_rcnn_m 结构）
    mmrotate_cfg='tools/cfg/lsk_s_fpn_1x_dota_le90.py',
    # E2E-MFD 训练好的权重（加载 ckpt['state_dict']）；先不管就留空，
    # 用 --mock-weights 可跳过加载跑通流程
    fusion_ckpt='work_dirs/lsk_s_fpn_1x_dota_le90/epoch_12.pth',

    # ---- 数据侧（测试集，visible 与 infrared 同名配对）----
    test_visible_dir=r'D:\我太想进步了\竞赛\人工智能算法挑战赛\data\测试集\visible',
    test_ir_dir=r'D:\我太想进步了\竞赛\人工智能算法挑战赛\data\测试集\infrared',

    # ---- 输出 ----
    # 融合图输出目录（自动创建；已存在的文件自动跳过 = 断点续跑）
    fusion_out_dir='fusion_test',
    # YOLOv8 结果输出根目录（下分 labels/ 与 images/）
    yolo_out_dir='yolo_out',

    # ---- YOLOv8 检测侧 ----
    # YOLOv8 权重：默认 yolov8n.pt（COCO 预训练，已下载到项目根，用于验证链路）；
    # 正式使用请换成自己训练好的 best.pt（比赛 12 类）。为空/不存在则跳过检测
    yolo_weights='yolov8n.pt',
    conf_thres=0.25,
    imgsz=640,
    device='auto',          # auto / cpu / cuda:0

    # 比赛 12 类（YOLO 检测输出类名，与标签映射一致）
    class_names=[
        'person', 'boat', 'animal', 'seat', 'sign', 'bicycle',
        'car', 'ball', 'light', 'garbage_can', 'uav', 'tricycle',
    ],
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LIMIT = 0  # 冒烟模式：只处理前 N 张（0=全部）


# ----------------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------------
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """支持中文路径的 cv2.imread（返回 BGR）"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def imwrite_unicode(path, img):
    """支持中文路径的 cv2.imwrite（img 为 BGR）"""
    ext = os.path.splitext(path)[1] or '.png'
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f'编码失败: {path}')
    buf.tofile(path)


def resolve_device(device):
    if device == 'auto':
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


# ----------------------------------------------------------------------------
# 阶段 A：E2E-MFD 融合网络生成融合图
# ----------------------------------------------------------------------------
def _patch_loss_device():
    """CPU 环境下 patch: 仓库 loss.py 的 LapLoss2 默认 device='cuda'（硬编码），
    构建 Oriented_rcnn_m 时会实例化 DetcropPixelLoss -> LapLoss2 -> .to('cuda') 崩溃。
    这里把默认 device 改成自动选择，不动仓库源码（GPU 环境行为不变）。"""
    import torch
    import mmrotate.models.detectors.loss as loss_mod

    _orig_init = loss_mod.LapLoss2.__init__

    def _patched(self, max_levels=3, channels=1, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _orig_init(self, max_levels=max_levels, channels=channels, device=device)

    loss_mod.LapLoss2.__init__ = _patched


def build_fusion_model(cfg_path, ckpt_path, device, mock_weights=False):
    """构建 Oriented_rcnn_m（含 backbone/neck/fusion），可选加载权重"""
    import torch
    from mmcv import Config
    from mmrotate.models.detectors.oriented_rcnn_m import Oriented_rcnn_m

    _patch_loss_device()
    cfg = Config.fromfile(os.path.join(PROJECT_ROOT, cfg_path))
    model = Oriented_rcnn_m(
        cfg.model.backbone, cfg.model.neck, cfg.model.rpn_head,
        cfg.model.roi_head, cfg.model.train_cfg, cfg.model.test_cfg)

    if not mock_weights:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f'未找到 E2E-MFD 权重: {ckpt_path}\n'
                f'  放好权重后重跑，或用 --mock-weights 以随机权重跑通流程验证。')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        model.load_state_dict(state, strict=False)
        print(f'[A] 已加载融合权重: {ckpt_path}')
    else:
        print('[A] 警告: 使用随机初始化权重（--mock-weights，仅验证流程）')

    model.eval().to(device)
    return model, cfg


def fuse_pair(model, cfg, visible_bgr, ir_bgr, device):
    """把一对 (visible, infrared) 融合成一张 RGB 图。

    复刻 oriented_rcnn_m.forward_fusion 的核心逻辑（去掉 DroneVehicle 尺寸裁剪）：
      F = w_ir * IR + w_rgb * RGB_V   （HSV 亮度域逐像素加权）
      融合亮度 V + RGB 的 H/S -> HSV -> RGB
    返回: (rgb_fusion, res_weight)  rgb_fusion 为 (H,W,3) uint8 RGB 顺序
    """
    import torch

    def to_t(x):
        return torch.from_numpy(x).float().to(device)

    # 1) RGB 的 HSV 分解（与 loading.py bri_clr_loader1 一致）
    hsv = cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2HSV)
    bri = hsv[:, :, 2]            # V 亮度 (H,W) 0-255
    clr = hsv[:, :, 0:2]          # H,S  (H,W,2) 0-255

    # 2) 模型输入张量
    vis_bri = to_t(bri[None, None])                     # (1,1,H,W) 0-255
    vis_clr = to_t(clr.transpose(2, 0, 1)[None])        # (1,2,H,W) 0-255
    f_rgb = to_t(visible_bgr.transpose(2, 0, 1)[None] / 255.0)  # (1,3,H,W) 0-1
    f_ir = to_t(ir_bgr.transpose(2, 0, 1)[None] / 255.0)         # (1,3,H,W) 0-1
    ir_img = f_ir[:, 0:1]                               # (1,1,H,W) 单通道红外

    # 3) backbone 输入的归一化（与 pipeline Normalize 一致，to_rgb=True）
    norm_cfg = cfg.img_norm_cfg
    od_rgb = to_t(
        cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        .transpose(2, 0, 1)[None])          # (1,3,H,W)
    od_ir = to_t(
        cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        .transpose(2, 0, 1)[None])          # (1,3,H,W)
    mean = to_t(np.array(norm_cfg['mean'], dtype=np.float32).reshape(1, 3, 1, 1))
    std = to_t(np.array(norm_cfg['std'], dtype=np.float32).reshape(1, 3, 1, 1))
    od_rgb = (od_rgb - mean) / std
    od_ir = (od_ir - mean) / std

    # 4) 融合前向：backbone 双流 -> 特征逐层相加 -> FusionNet 权重图
    with torch.no_grad():
        x_rgb = model.backbone(od_rgb)
        x_ir = model.backbone(od_ir)
        if model.with_neck:
            x_rgb = model.neck(x_rgb)
            x_ir = model.neck(x_ir)
        features = [x_rgb[i] + x_ir[i] for i in range(4)]
        inputs = torch.cat([ir_img, f_rgb], dim=1)      # (1,4,H,W)
        _, res_weight = model.fusion(features, inputs)  # (1,2,H,W) sigmoid
        fus_bri = res_weight[:, 0:1] * ir_img + res_weight[:, 1:2] * vis_bri

    # 5) 颜色恢复（与 change_hsv2rgb 一致）：融合亮度 + RGB 的 H/S -> HSV -> RGB
    bri_np = fus_bri[0, 0].cpu().numpy()
    mn, mx = bri_np.min(), bri_np.max()
    scale = 255.0 / max(mx - mn, 1e-6)
    bri_np = np.clip((bri_np - mn) * scale, 0, 255).astype(np.uint8)
    hsv_f = np.concatenate(
        [vis_clr[0].cpu().numpy().transpose(1, 2, 0), bri_np[..., None]], axis=2)
    hsv_f = hsv_f.astype(np.uint8)
    hsv_f[:, :, 2] = bri_np
    rgb = cv2.cvtColor(hsv_f, cv2.COLOR_HSV2RGB)        # RGB 顺序
    return rgb, res_weight


def fuse_test_set(model, cfg, device):
    """遍历测试集生成融合图，已存在的跳过（断点续跑）"""
    vis_dir = CONFIG['test_visible_dir']
    ir_dir = CONFIG['test_ir_dir']
    out_dir = os.path.join(PROJECT_ROOT, CONFIG['fusion_out_dir'])
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(vis_dir) or not os.path.isdir(ir_dir):
        raise FileNotFoundError(f'测试集目录不存在: {vis_dir} / {ir_dir}')

    names = sorted(os.listdir(vis_dir))
    if LIMIT:
        names = names[:LIMIT]
    if not names:
        raise FileNotFoundError(f'{vis_dir} 为空')
    print(f'[A] 测试集 {len(names)} 张，融合输出 -> {out_dir}')

    done = skip = 0
    for name in names:
        out_path = os.path.join(out_dir, os.path.splitext(name)[0] + '.png')
        if os.path.exists(out_path):
            skip += 1
            continue
        vis_path = os.path.join(vis_dir, name)
        ir_path = os.path.join(ir_dir, name)
        if not os.path.exists(ir_path):
            print(f'[A] 跳过（缺红外配对）: {name}')
            continue
        visible_bgr = imread_unicode(vis_path)
        ir_bgr = imread_unicode(ir_path)
        if visible_bgr is None or ir_bgr is None:
            print(f'[A] 读取失败: {name}')
            continue
        if visible_bgr.shape[:2] != ir_bgr.shape[:2]:
            ir_bgr = cv2.resize(ir_bgr, visible_bgr.shape[:2][::-1])
        rgb, _ = fuse_pair(model, cfg, visible_bgr, ir_bgr, device)
        imwrite_unicode(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        done += 1
        if (done + skip) % 100 == 0:
            print(f'[A] 进度 {done + skip}/{len(names)} (新增 {done}, 跳过 {skip})')
    print(f'[A] 融合完成: 新增 {done} 张, 跳过已有 {skip} 张')
    return out_dir


# ----------------------------------------------------------------------------
# 阶段 B：YOLOv8 检测
# ----------------------------------------------------------------------------
def detect_with_yolov8(fusion_dir):
    """对融合图目录做 YOLOv8 检测，输出 labels/*.txt（YOLO 归一化格式）+ 可视化图"""
    weights = CONFIG['yolo_weights']
    if not weights:
        print('[B] yolo_weights 未配置，跳过 YOLOv8 检测（融合图已生成）')
        return
    if not os.path.exists(weights):
        print(f'[B] 未找到 YOLOv8 权重: {weights}，跳过检测（融合图已生成）')
        return

    from ultralytics import YOLO

    yolo = YOLO(weights)
    txt_dir = os.path.join(PROJECT_ROOT, CONFIG['yolo_out_dir'], 'labels')
    vis_dir = os.path.join(PROJECT_ROOT, CONFIG['yolo_out_dir'], 'images')
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    device = resolve_device(CONFIG['device'])
    print(f'[B] YOLOv8 检测: {weights} @ {device}, conf={CONFIG["conf_thres"]}')

    names = sorted(os.listdir(fusion_dir))
    if LIMIT:
        names = names[:LIMIT]
    done = skip = 0
    for name in names:
        stem = os.path.splitext(name)[0]
        txt_path = os.path.join(txt_dir, stem + '.txt')
        if os.path.exists(txt_path):
            skip += 1
            continue
        img_path = os.path.join(fusion_dir, name)
        res = yolo.predict(
            source=img_path, conf=CONFIG['conf_thres'], imgsz=CONFIG['imgsz'],
            device=device, verbose=False)[0]
        # 自己写 YOLO 归一化格式标签: cls cx cy w h（不依赖 ultralytics save_txt API）
        with open(txt_path, 'w', encoding='utf-8') as f:
            if res.boxes is not None:
                cls = res.boxes.cls.cpu().numpy().astype(int)
                xywhn = res.boxes.xywhn.cpu().numpy()
                for c, (cx, cy, w, h) in zip(cls, xywhn):
                    f.write(f'{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        imwrite_unicode(os.path.join(vis_dir, stem + '.png'), res.plot())
        done += 1
        if (done + skip) % 100 == 0:
            print(f'[B] 进度 {done + skip}/{len(names)} (新增 {done}, 跳过 {skip})')
    print(f'[B] 检测完成: 新增 {done} 张, 跳过已有 {skip} 张 -> {txt_dir}')
    return txt_dir


# ----------------------------------------------------------------------------
# 环境自检
# ----------------------------------------------------------------------------
def run_check():
    print('== 依赖检查 ==')
    import torch
    import torchvision
    import cv2
    import numpy as np
    print(f'  torch {torch.__version__} / torchvision {torchvision.__version__}')
    print(f'  numpy {np.__version__} / opencv {cv2.__version__}')
    assert np.__version__.startswith('1.'), 'numpy 必须 <2 (mmcv 1.7.2 要求)'
    import mmcv
    import mmdet
    import mmrotate
    import timm
    import e2cnn
    print(f'  mmcv {mmcv.__version__} / mmdet {mmdet.__version__} / mmrotate {mmrotate.__version__}')
    print(f'  timm {timm.__version__} / e2cnn ok')
    try:
        import ultralytics
        print(f'  ultralytics {ultralytics.__version__}')
    except ImportError:
        print('  ultralytics 未安装: 请执行 pip install ultralytics')

    print('== 路径检查 ==')
    for k in ('test_visible_dir', 'test_ir_dir'):
        p = CONFIG[k]
        print(f'  {k}: {p} -> {"OK" if os.path.isdir(p) else "缺失!"}')

    print('== 权重检查（可先不管）==')
    for k in ('fusion_ckpt', 'yolo_weights'):
        p = CONFIG[k]
        if not p:
            print(f'  {k}: 未配置（占位）')
        elif os.path.exists(p):
            print(f'  {k}: OK -> {p}')
        else:
            print(f'  {k}: 未找到 -> {p} (可先用 --mock-weights 验证流程)')


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='E2E-MFD 融合图 + YOLOv8 检测')
    ap.add_argument('--fuse-only', action='store_true', help='只生成融合图')
    ap.add_argument('--detect-only', action='store_true', help='只用已有融合图检测')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 张（0=全部）')
    ap.add_argument('--mock-weights', action='store_true',
                    help='不加载 E2E-MFD 权重（随机权重跑通流程）')
    ap.add_argument('--check', action='store_true', help='只检查环境与路径')
    args = ap.parse_args()

    if args.check:
        run_check()
        return

    if args.limit:
        global LIMIT
        LIMIT = args.limit
        print(f'[i] 冒烟模式: 只处理前 {LIMIT} 张')

    device = resolve_device(CONFIG['device'])
    print(f'[i] device={device}')

    if not args.detect_only:
        model, cfg = build_fusion_model(
            CONFIG['mmrotate_cfg'], CONFIG['fusion_ckpt'], device,
            mock_weights=args.mock_weights)
        fusion_dir = fuse_test_set(model, cfg, device)
    else:
        fusion_dir = os.path.join(PROJECT_ROOT, CONFIG['fusion_out_dir'])
        if not os.path.isdir(fusion_dir):
            raise FileNotFoundError(f'融合图目录不存在: {fusion_dir}，先跑融合阶段')
        print(f'[i] 复用已有融合图: {fusion_dir}')

    if not args.fuse_only:
        detect_with_yolov8(fusion_dir)


if __name__ == '__main__':
    main()

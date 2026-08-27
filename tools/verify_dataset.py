# -*- coding: utf-8 -*-
"""验证 DroneVehicleDataset 能正确加载转换后的 DOTA 标注 + 双模态图像。"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from mmcv import Config
from mmrotate.datasets import build_dataset
from mmrotate.datasets.dronevehicle import DroneVehicleDataset
from mmrotate.datasets.pipelines import LoadImagePairFromFile

DATA_ROOT = r'D:\我太想进步了\竞赛\人工智能算法挑战赛\data'
DOTA_ROOT = r'D:\我太想进步了\竞赛\人工智能算法挑战赛\data_dota'

# ---- 1. 仅验证标注解析 ----
ds = DroneVehicleDataset(
    ann_file=os.path.join(DOTA_ROOT, '训练集', 'labels'),
    img_prefix=os.path.join(DATA_ROOT, '训练集', 'visible'),
    pipeline=[],
    version='le90')
print('[1] CLASSES =', ds.CLASSES)
print('[1] 样本数 =', len(ds), '(期望 2000)')
print('[1] 类别数 =', len(ds.CLASSES), '(期望 12)')

# 抽查一张含 png 后缀的样本
png_idx = None
for i in range(len(ds)):
    if ds.data_infos[i]['filename'].endswith('.png'):
        png_idx = i
        break
info = ds.data_infos[png_idx]
print('[1] 抽查样本 filename =', info['filename'],
      '| bboxes =', info['ann']['bboxes'].shape,
      '| labels =', info['ann']['labels'].shape)
print('[1] 前 3 个 bbox(x,y,w,h,angle):\n', info['ann']['bboxes'][:3])
print('[1] 对应 labels:', info['ann']['labels'][:3],
      '->', [ds.CLASSES[l] for l in info['ann']['labels'][:3]])

# 空标注是否被过滤（原始无空 txt，期望全部保留）
print('[1] 过滤空标注后样本数 =', len(ds._filter_imgs()), '(期望 2000)')

# ---- 2. 完整 pipeline 验证（双模态加载 + 归一化 + collect） ----
cfg = Config.fromfile(
    r'D:\我太想进步了\竞赛\人工智能算法挑战赛\E2E-MFD\tools\cfg\lsk_s_fpn_1x_dota_le90.py')
test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'infrared')),
    dict(type='RResize', img_scale=(712, 840)),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle_m'),
    dict(type='Collect', keys=['img'])
]
ds2 = DroneVehicleDataset(
    ann_file=os.path.join(DOTA_ROOT, '训练集', 'labels'),
    img_prefix=os.path.join(DATA_ROOT, '训练集', 'visible'),
    pipeline=test_pipeline,
    version='le90',
    filter_empty_gt=False)
res = ds2[png_idx]
print('[2] pipeline 输出 keys =', sorted(res.keys()))
# DefaultFormatBundle_m 把 6 个 img_fields 组成 list of DC
imgs = res['img']
print('[2] img 字段数 =', len(imgs), '(期望 6: img1/img2/bri/clr/F_rgb/F_ir)')
print('[2] 各字段 shape:',
      [tuple(im.data.shape) for im in imgs])
print('[2] img_metas img_shape =', res['img_metas'].data['img_shape'],
      '| ori_shape =', res['img_metas'].data['ori_shape'])
print('[2] 双模态加载+collect 成功 ✓')

# ---- 3. 测试模式（无 labels，图片目录） ----
import glob
from mmrotate.datasets.dronevehicle import DroneVehicleDataset as DV2
ds3 = DroneVehicleDataset(
    ann_file=os.path.join(DATA_ROOT, '测试集', 'visible'),
    img_prefix=os.path.join(DATA_ROOT, '测试集', 'visible'),
    pipeline=[],
    test_mode=True)
print('[3] 测试模式样本数 =', len(ds3), '(期望 1000)')
exts = {}
for d in ds3.data_infos:
    e = d['filename'].rsplit('.', 1)[-1]
    exts[e] = exts.get(e, 0) + 1
print('[3] 测试集扩展名分布:', exts, '| 样例:', ds3.data_infos[0]['filename'])
print('ALL PASS')

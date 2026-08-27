# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class DroneVehicleDataset(CustomDataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    # 12 类比赛类别定义（0-11），类别名无空格（DOTA 按空白分列）
    CLASSES = ('person', 'boat', 'animal', 'seat', 'sign', 'bicycle', 'car',
               'ball', 'light', 'garbage_can', 'uav', 'tricycle')

    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty

        super(DroneVehicleDataset, self).__init__(ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            # 支持混合扩展名（jpg/png），文件名保留真实扩展名
            ann_files = []
            for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                ann_files += glob.glob(ann_folder + '/*' + ext)
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = osp.split(ann_file)[1]
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                # 图片扩展名可能为 jpg/png 混合，探测真实扩展名
                img_name = self._find_img(img_id)
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    lines = [l for l in f.readlines() if l.strip()]
                if not lines:
                    continue

                # ============ 统一转为 DOTA v1 行 ============
                # 5 列  -> YOLO 格式(cls_id cx cy w h，归一化) -> 转 DOTA
                # >=10 列 -> 本身就是 DOTA 格式(x1 y1 ... x4 y4 cls diff)
                n_col = len(lines[0].split())
                if n_col == 5:
                    # 反归一化需要图片真实尺寸
                    img_w, img_h = self._get_img_size(img_name)
                    dota_lines = []
                    for si in lines:
                        parts = si.split()
                        cls_id = int(parts[0])
                        if not 0 <= cls_id < len(self.CLASSES):
                            raise ValueError(
                                f'Invalid class id {cls_id} in {ann_file}')
                        cls_name = self.CLASSES[cls_id]
                        cx = float(parts[1]) * img_w
                        cy = float(parts[2]) * img_h
                        bw = float(parts[3]) * img_w
                        bh = float(parts[4]) * img_h
                        # 水平框 -> 4 角点（顺时针），难度填 0
                        x1, y1 = cx - bw / 2, cy - bh / 2
                        x2, y2 = cx + bw / 2, cy + bh / 2
                        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        dota_lines.append(
                            ' '.join(f'{p:.2f}' for xy in pts for p in xy) +
                            f' {cls_name} 0')
                elif n_col >= 10:
                    dota_lines = lines
                else:
                    raise ValueError(
                        f'Unknown annotation format in {ann_file}: '
                        f'{n_col} columns, first line: '
                        f'{lines[0].strip() if lines else "(empty)"}')

                # ============ 统一按 DOTA v1 解析 ============
                for si in dota_lines:
                    bbox_info = si.split()
                    poly = np.array(bbox_info[:8], dtype=np.float32)
                    try:
                        x, y, w, h, a = poly2obb_np(poly, self.version)
                    except:  # noqa: E722
                        continue
                    cls_name = bbox_info[8]
                    difficulty = int(bbox_info[9])
                    label = cls_map[cls_name]
                    if difficulty <= self.difficulty:
                        gt_bboxes.append([x, y, w, h, a])
                        gt_labels.append(label)
                        gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def _find_img(self, img_id):
        """按 id 探测 img_prefix 下真实图片文件名（jpg/png 混合兼容）。"""
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            p = osp.join(self.img_prefix, img_id + ext)
            if osp.exists(p):
                return img_id + ext
        # 兜底：找不到时回退 jpg，让后续加载器报错定位问题
        return img_id + '.jpg'

    def _get_img_size(self, img_name):
        """按文件名读取图片宽高，用于 YOLO 归一化坐标反算像素。

        Returns:
            tuple: (img_w, img_h)
        """
        p = osp.join(self.img_prefix, img_name)
        try:
            from PIL import Image
            with Image.open(p) as im:
                return im.size  # (w, h)
        except Exception:  # noqa: E722
            import cv2
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(f'Cannot read image: {p}')
            return img.shape[1], img.shape[0]

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            if x_y:  # DOTA 切图命名: ori__x___y
                x_y_2 = re.findall(r'\d+', x_y[0])
                x, y = int(x_y_2[0]), int(x_y_2[1])
            else:  # 整图命名（非切图），偏移为 0
                x, y = 0, 0
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)

    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Args:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if osp.exists(out_folder):
            raise ValueError(f'The out_folder should be a non-exist path, '
                             f'but {out_folder} is existing')
        os.makedirs(out_folder)

        files = [
            osp.join(out_folder, 'Task1_' + cls + '.txt')
            for cls in self.CLASSES
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    txt_element = [img_id, str(bbox[-1])
                                   ] + [f'{p:.2f}' for p in bbox[:-1]]
                    f.writelines(' '.join(txt_element) + '\n')

        for f in file_objs:
            f.close()

        target_name = osp.split(out_folder)[-1]
        with zipfile.ZipFile(
                osp.join(out_folder, target_name + '.zip'), 'w',
                zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return files

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            submission_dir (str, optional): The folder that contains submission
                files. If not specified, a temp folder will be created.
                Default: None.
            nproc (int, optional): number of process.

        Returns:
            tuple:

                - result_files (dict): a dict containing the json filepaths
                - tmp_dir (str): the temporal directory created for saving \
                    json files when submission_dir is not specified.
        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        id_list, dets_list = self.merge_det(results, nproc)
        stop_time = time.time()
        print(f'Used time: {(stop_time - start_time):.1f} s')

        result_files = self._results2submission(id_list, dets_list,
                                                submission_dir)

        return result_files, tmp_dir


def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results

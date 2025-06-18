from PIL import Image
import time
import numpy as np
import torch
import os,sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from lvis.eval import LVISEval
sys.path.append("..")
import pycocotools.mask as mask_util
import json
import argparse
from tqdm import tqdm
from torchvision.transforms import ToTensor
# import warnings
# from utils.eval_load import SA1BDataset, COCODataset
# from mmengine.dataset import pseudo_collate, worker_init_fn, DefaultSampler
# from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
#                            is_distributed)
# from mmengine.utils import ProgressBar
# from mmdet.structures.bbox import bbox_overlaps
# from functools import partial
# from torch.utils.data import DataLoader
from torch.nn import functional as F
import random
from utils.common import (cal_iou, sample_point_in_mask, xywh2xyxy, xyxy2xywh, make_fig, sample_prompts, get_centroid_from_mask)
from typing import Optional
import multiprocessing
import pickle

# 忽略所有警告
# warnings.filterwarnings("ignore")

def get_points(point_from, gt_mask, boxes, num_points:Optional=1, device:Optional='cuda:0'):

    # gt_mask_1 = torch.tensor(gt_mask).unsqueeze(0).unsqueeze(0)
    # for g_1 in gt_mask_1.squeeze(1):
    #     candidate_indices_1 = g_1.nonzero()
    #     try:
    #         selected_index_1 = random.randint(0, len(candidate_indices_1) - 1)
    #     except:
    #         print("done")

    if point_from == 'mask-rand':
        gt_mask = torch.tensor(gt_mask, device=device).unsqueeze(0).unsqueeze(0)
        point_list = []
        for i in range(num_points):
            for g in gt_mask.squeeze(1):
                candidate_indices = g.nonzero()
                selected_index = random.randint(0, len(candidate_indices) - 1)
                p = candidate_indices[selected_index].flip(0)
                point_list.append(p)
        points = torch.stack(point_list, dim=0)
        labels = torch.ones(points.shape[0])
        points = (points.detach().cpu().numpy(), labels.detach().cpu().numpy())

    elif point_from == 'box-center':
        boxes = torch.tensor(boxes, device=device)
        center_x = (boxes[0] + boxes[2]) / 2
        center_y = (boxes[1] + boxes[3]) / 2
        boxes = None
        points = np.array([[center_x.item(), center_y.item()]])
        labels = torch.ones(points.shape[0])
        points = (points, labels.numpy())

    elif point_from == 'mask-center':
        # gt_mask = torch.tensor(gt_mask).unsqueeze(0).unsqueeze(0)
        gt_mask = torch.tensor(gt_mask, device=device).unsqueeze(0).unsqueeze(0)
        points = get_centroid_from_mask(gt_mask > 0.5)
        labels = torch.ones(points.shape[:2])
        # points = (points[0].numpy(), labels[0].numpy())
        points = (points[0].detach().cpu().numpy(), labels[0].detach().cpu().numpy())
    return points


def calculate_iou(mask1, mask2):
    """
    计算两个二值掩码的 IoU 值。

    参数:
    mask1: np.ndarray, 第一个掩码 (二值数组，0 和 1 表示无和有掩码)
    mask2: np.ndarray, 第二个掩码 (二值数组，0 和 1 表示无和有掩码)

    返回:
    float: IoU 值
    """
    # 计算交集和并集
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # 避免除以零的情况
    if union == 0:
        return 0.0

    # 计算 IoU
    iou = intersection / union
    return iou


def show_mask(origin_image, mask, points_prompt:Optional=None):
    # 显示原始图像
    # plt.figure(figsize=(10, 10))
    # plt.imshow(origin_image)
    # plt.axis('off')
    # plt.title("Original Image")
    # plt.show()

    # 显示掩码
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray', alpha=0.5)  # 使用灰度显示掩码
    plt.axis('off')
    plt.title("Segmentation Mask")
    plt.show()

    if points_prompt is None:
        # 将掩码叠加到原图上
        plt.figure(figsize=(10, 10))
        plt.imshow(origin_image)
        plt.imshow(mask, cmap='jet', alpha=0.5)  # 使用伪彩色掩码叠加
        plt.axis('off')
        plt.title("Image with Mask Overlay")
        plt.show()

    else:
        # 将掩码叠加到原图上
        plt.figure(figsize=(10, 10))
        plt.imshow(origin_image)
        plt.imshow(mask, cmap='jet', alpha=0.5)  # 使用伪彩色掩码叠加
        for i in points_prompt:
            plt.scatter(i[0], i[1], c='red', s=100, marker='*', label="Point")
        plt.axis('off')
        plt.title("Image with Mask Overlay")
        plt.show()



def point_in_boxes(point, boxes, scores):
    selected = (boxes[:, 0] < point[0]) & \
               (point[0] < boxes[:, 2]) & \
               (boxes[:, 1] < point[1]) & \
               (point[1] < boxes[:, 3])
    return boxes[selected], scores[selected]


def point_box_dist(point, boxes):
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2
    center = torch.stack([center_x, center_y], dim=1)
    point = point[None].expand(boxes.shape[0], 2)
    return F.pairwise_distance(point, center)

def k_nearest_center(point, boxes, scores, topk=1):
    distance = -point_box_dist(point, boxes)
    selected = distance.topk(min(topk, boxes.shape[0]), dim=0)[1]
    return boxes[selected], scores[selected]


def run_ours_box_or_points(img_path, pts_sampled, pts_labels, model):
    image_np = np.array(Image.open(img_path))
    img_tensor = ToTensor()(image_np)
    pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
    pts_labels = torch.reshape(torch.tensor(pts_labels), [1, 1, -1])
    predicted_logits, predicted_iou = model(
        img_tensor[None, ...],
        pts_sampled,
        pts_labels,
    )

    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )
    # masks = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
    return torch.ge(predicted_logits[0, 0, :], 0).cpu().detach().numpy(), predicted_iou[0,0,:].detach().numpy()


def eval_coco_box(model_name, predictor, eval_type, root_path, val_img_path, val_json_path, vit_det_file_path):

    val_img_path = root_path + val_img_path
    val_json_path = root_path + val_json_path
    vit_det_file_path = root_path + vit_det_file_path

    with open(vit_det_file_path) as f:
        res = json.load(f)

    pre_img_id = 0
    print("Total instances num:", len(res))
    Estimated_mIoU = 0
    mIoU = 0
    len_res = len(res)
    error = 0
    for i in tqdm(range(len(res))):
        res_ins = res[i]

        # mask_label
        gt_mask = mask_util.decode(res[i]["segmentation"])
        img_id = res_ins['image_id']
        img_file_name = f'{img_id:012d}' + '.jpg'

        if model_name in ['efficient_sam_ti', 'efficient_sam_s']:
            input_box = res_ins['bbox']
            input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            input_point = np.reshape(np.array(input_box), (-1,2))
            input_label = np.array([2, 3])

            if pre_img_id != img_id:
                image_np = np.array(Image.open(val_img_path + img_file_name))
                img_tensor = ToTensor()(image_np)
                predictor.set_image(img_tensor[None, ...])
                pre_img_id = img_id

            if image_np.shape[:-1] != gt_mask.shape:
                len_res -= 1
                error += 1
                continue

            masks, ious = run_ours_box_or_points(val_img_path + img_file_name,
                                                 input_point,
                                                 input_label,
                                                 predictor
                                                 )

            # predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            # show the masks #
            # show_mask(origin_image=image, mask=gt_mask)
            # show_mask(origin_image=image, mask=pred_mask)
            # print(calculate_iou(gt_mask, pred_mask))

            mIoU += calculate_iou(gt_mask, pred_mask)
            Estimated_mIoU += max(ious)

            new_seg = mask_util.encode(np.array(masks[np.argmax(ious)], order="F", dtype="uint8"))
            new_seg["counts"] = new_seg["counts"].decode("utf-8")
            res[i]["segmentation"] = new_seg

        else:
            if pre_img_id != img_id:
                image = cv2.imread(val_img_path + img_file_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                predictor.set_image(image)
                pre_img_id = img_id

            if image.shape[:-1] != gt_mask.shape:
                len_res -= 1
                error += 1
                continue

            input_box = res_ins['bbox']
            input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            input_box = np.array(input_box)
            masks, ious, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
            )

            # predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            # show the masks #
            # show_mask(origin_image=image, mask=gt_mask)
            # show_mask(origin_image=image, mask=pred_mask)
            # print(calculate_iou(gt_mask, pred_mask))

            mIoU += calculate_iou(gt_mask, pred_mask)
            Estimated_mIoU += max(ious)

            new_seg=mask_util.encode(np.array(masks[np.argmax(ious)],order="F", dtype="uint8"))
            new_seg["counts"] = new_seg["counts"].decode("utf-8")
            res[i]["segmentation"]=new_seg

    print(f"Finally total instances num: {len_res} , because there are {error} destroyed gt_masks")
    print(f"Estimated_mIoU : {Estimated_mIoU / len_res}")
    print(f"mIoU : {mIoU / len_res}")
    results = {
        'len_res' : len_res,
        'error' : error,
        'Estimated_mIoU' : Estimated_mIoU / len_res,
        'mIoU' : mIoU / len_res
    }

    for c in res:
        c.pop("bbox", None)

    cocoGT = COCO(val_json_path)
    coco_dt = cocoGT.loadRes(res)
    coco_eval = COCOeval(cocoGT, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return results


def eval_coco_box_perimage(eval_input): # (model_name, res_i_list, predictor, img_id, val_img_path)
    model_name = eval_input[0]
    res_i_list = eval_input[1]
    predictor = eval_input[2]
    img_id = eval_input[3]
    val_img_path = eval_input[4]

    img_file_name = f'{img_id:012d}' + '.jpg'
    image = cv2.imread(val_img_path + img_file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    numbers = len(res_i_list)
    miou = 0
    for res_ins in res_i_list:
        input_box = res_ins['bbox']
        gt_mask = mask_util.decode(res_ins["segmentation"])
        input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
        input_box = np.array(input_box)
        masks, ious, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
        )

        # predict_mask
        pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

        miou += calculate_iou(gt_mask, pred_mask)

    return miou



def eval_lvis_box_perimage(eval_input): # (model_name, res_i_list, predictor, img_id, val_img_path)
    model_name = eval_input[0]
    res_i_list = eval_input[1]
    predictor = eval_input[2]
    img_id = eval_input[3]
    val_img_path = eval_input[4]

    img_file_name = f'{img_id:012d}' + '.jpg'
    image = cv2.imread(val_img_path + img_file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    miou = 0
    error = 0
    for res_ins in res_i_list:
        if model_name in ['efficient_sam_ti', 'efficient_sam_s']:

            input_box = res_ins['bbox']
            input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            input_point = np.reshape(np.array(input_box), (-1, 2))
            input_label = np.array([2, 3])
            gt_mask = mask_util.decode(res_ins["segmentation"])

            # if pre_img_id != img_id:
            #     image_np = np.array(Image.open(val_img_path + img_file_name))
            #     img_tensor = ToTensor()(image_np)
            #     predictor.set_image(img_tensor[None, ...])
            #     pre_img_id = img_id

            if image.shape[:-1] != gt_mask.shape:
                error += 1

            masks, ious = run_ours_box_or_points(val_img_path + img_file_name,
                                                 input_point,
                                                 input_label,
                                                 predictor
                                                 )

            # predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            # show the masks #
            # show_mask(origin_image=image, mask=gt_mask)
            # show_mask(origin_image=image, mask=pred_mask)
            # print(calculate_iou(gt_mask, pred_mask))

            miou += calculate_iou(gt_mask, pred_mask)

        else:
            input_box = res_ins['bbox']
            gt_mask = mask_util.decode(res_ins["segmentation"])
            input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            input_box = np.array(input_box)

            if image.shape[:-1] != gt_mask.shape:
                error += 1

            masks, ious, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
            )

            # predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            miou += calculate_iou(gt_mask, pred_mask)

    return [miou, error]


def eval_lvis_point_perimage(eval_input): # (model_name, res_i_list, predictor, img_id, val_img_path, point_from)
    model_name = eval_input[0]
    res_i_list = eval_input[1]
    predictor = eval_input[2]
    img_id = eval_input[3]
    val_img_path = eval_input[4]
    point_from = eval_input[5]
    num_points = eval_input[6]


    img_file_name = f'{img_id:012d}' + '.jpg'

    if model_name in ['efficient_sam_ti', 'efficient_sam_s']:
        image_np = np.array(Image.open(val_img_path+img_file_name))
        img_tensor = ToTensor()(image_np)
        predictor.set_image(img_tensor[None, ...])
    else:
        image = cv2.imread(val_img_path + img_file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)


    error = 0
    mIoU = 0
    n_items = 0
    for res_ins in res_i_list:
        if model_name in ['efficient_sam_ti', 'efficient_sam_s']:
            input_box = res_ins['bbox']
            gt_mask = mask_util.decode(res_ins["segmentation"])
            # input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            # input_point = np.reshape(np.array(input_box), (-1, 2))
            # input_label = np.array([2, 3])

            # if pre_img_id != img_id:
            #     image_np = np.array(Image.open(val_img_path + img_file_name))
            #     img_tensor = ToTensor()(image_np)
            #     predictor.set_image(img_tensor[None, ...])
            #     pre_img_id = img_id


            prompt_points = res_ins['prompt']
            if prompt_points == None:
                error += 1
                continue

            # if point_from == 'mask-rand':
            #     prompt_points = get_points(point_from, gt_mask, input_box, num_points=num_points)
            # else:
            #     prompt_points = get_points(point_from, gt_mask, input_box)

            masks, ious = run_ours_box_or_points(val_img_path + img_file_name,
                                                 prompt_points[0],
                                                 prompt_points[1],
                                                 predictor
                                                 )

            # mask_label & predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            # show the masks #
            # show_mask(origin_image=image_np, mask=seg_label)
            # show_mask(origin_image=image_np, mask=pred_mask, points_prompt=prompt_points[0])
            # print(calculate_iou(seg_label, pred_mask))

            mIoU += calculate_iou(gt_mask, pred_mask)
            n_items += 1

            # new_seg = mask_util.encode(np.array(masks[np.argmax(ious)], order="F", dtype="uint8"))
            # new_seg["counts"] = new_seg["counts"].decode("utf-8")
            # res_ins["segmentation"] = new_seg

        else:
            gt_mask = mask_util.decode(res_ins["segmentation"])
            # input_box = res_ins['bbox']
            # input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            # input_box = np.array(input_box)

            # if pre_img_id != img_id:
            #     image = cv2.imread(val_img_path + img_file_name)
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #     predictor.set_image(image)
            #     pre_img_id = img_id

            # if image.shape[:-1] != gt_mask.shape or torch.tensor(gt_mask).nonzero().numel() == 0:
            #     error += 1
            #     continue

            prompt_points = res_ins['prompt']
            if prompt_points == None:
                error += 1
                continue


            # if point_from == 'mask-rand':
            #     prompt_points = get_points(point_from, gt_mask, input_box, num_points=num_points)
            # else:
            #     prompt_points = get_points(point_from, gt_mask, input_box)

            masks, ious, _ = predictor.predict(
                point_coords=prompt_points[0],
                point_labels=prompt_points[1],
                box=None,
            )

            # mask_label & predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            # # show the masks #
            # show_mask(origin_image=image, mask=gt_mask)
            # show_mask(origin_image=image, mask=pred_mask, points_prompt=prompt_points[0])
            # print(calculate_iou(gt_mask, pred_mask))

            mIoU += calculate_iou(gt_mask, pred_mask)
            n_items += 1

            # new_seg = mask_util.encode(np.array(masks[np.argmax(ious)], order="F", dtype="uint8"))
            # new_seg["counts"] = new_seg["counts"].decode("utf-8")
            # res[i]["segmentation"] = new_seg

    return [mIoU, n_items]






def eval_coco_box_multi(model_name, predictor, eval_type, root_path, val_img_path, val_json_path, vit_det_file_path, num_cores):

    val_img_path = root_path + val_img_path
    val_json_path = root_path + val_json_path
    vit_det_file_path = root_path + vit_det_file_path

    with open(vit_det_file_path) as f:
        res = json.load(f)

    res_multi_dict = {}
    for i in tqdm(range(len(res))):
        res_ins = res[i]
        img_id = res_ins['image_id']
        if img_id not in res_multi_dict:
            res_multi_dict[img_id] = [res_ins]
        else:
            res_multi_dict[img_id].append(res_ins)


    eval_input = [(model_name, res_i_list, predictor, img_id, val_img_path) for img_id, res_i_list in res_multi_dict.items()]
    with multiprocessing.Pool(processes=num_cores) as pool:
        sum_miou_list = list(tqdm(pool.imap(eval_coco_box_perimage, eval_input), total=len(eval_input), desc='evaling'))

    print(np.sum(sum_miou_list)/len(res))

    #######################################################################
    pre_img_id = 0
    print("Total instances num:", len(res))
    Estimated_mIoU = 0
    mIoU = 0
    len_res = len(res)
    error = 0
    for i in tqdm(range(len(res))):
        res_ins = res[i]

        # mask_label
        gt_mask = mask_util.decode(res[i]["segmentation"])
        img_id = res_ins['image_id']
        img_file_name = f'{img_id:012d}' + '.jpg'

        if model_name in ['efficient_sam_ti', 'efficient_sam_s']:
            input_box = res_ins['bbox']
            input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            input_point = np.reshape(np.array(input_box), (-1,2))
            input_label = np.array([2, 3])

            if pre_img_id != img_id:
                image_np = np.array(Image.open(val_img_path + img_file_name))
                img_tensor = ToTensor()(image_np)
                predictor.set_image(img_tensor[None, ...])
                pre_img_id = img_id

            if image_np.shape[:-1] != gt_mask.shape:
                len_res -= 1
                error += 1
                continue

            masks, ious = run_ours_box_or_points(val_img_path + img_file_name,
                                                 input_point,
                                                 input_label,
                                                 predictor
                                                 )

            # predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            # show the masks #
            # show_mask(origin_image=image, mask=gt_mask)
            # show_mask(origin_image=image, mask=pred_mask)
            # print(calculate_iou(gt_mask, pred_mask))

            mIoU += calculate_iou(gt_mask, pred_mask)
            Estimated_mIoU += max(ious)

            # if i == 200:
            #     print(f"IoU(i==200) : {mIoU/200}")
            #     print("stop")
            new_seg = mask_util.encode(np.array(masks[np.argmax(ious)], order="F", dtype="uint8"))
            new_seg["counts"] = new_seg["counts"].decode("utf-8")
            res[i]["segmentation"] = new_seg

        else:
            if pre_img_id != img_id:
                image = cv2.imread(val_img_path + img_file_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                predictor.set_image(image)
                pre_img_id = img_id

            if image.shape[:-1] != gt_mask.shape:
                len_res -= 1
                error += 1
                continue

            input_box = res_ins['bbox']
            input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            input_box = np.array(input_box)
            masks, ious, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
            )

            # predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            # show the masks #
            # show_mask(origin_image=image, mask=gt_mask)
            # show_mask(origin_image=image, mask=pred_mask)
            # print(calculate_iou(gt_mask, pred_mask))

            mIoU += calculate_iou(gt_mask, pred_mask)
            Estimated_mIoU += max(ious)
            # if i == 200:
            #     print(f"mIoU(i==200) : {mIoU/200}")
            #     print("stop")
            new_seg=mask_util.encode(np.array(masks[np.argmax(ious)],order="F", dtype="uint8"))
            new_seg["counts"] = new_seg["counts"].decode("utf-8")
            res[i]["segmentation"]=new_seg

    print(f"Finally total instances num: {len_res} , because there are {error} destroyed gt_masks")
    print(f"Estimated_mIoU : {Estimated_mIoU / len_res}")
    print(f"mIoU : {mIoU / len_res}")

    results = {
        'len_res' : len_res,
        'error' : error,
        'Estimated_mIoU' : Estimated_mIoU / len_res,
        'mIoU' : mIoU / len_res
    }

    for c in res:
        c.pop("bbox", None)

    cocoGT = COCO(val_json_path)
    coco_dt = cocoGT.loadRes(res)
    coco_eval = COCOeval(cocoGT, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return results


def eval_lvis_box(model_name, predictor, eval_type, root_path, val_img_path, val_json_path, vit_det_file_path, val_lvis_instances_results_vitdet):

    val_img_path = root_path + val_img_path
    val_json_path = root_path + val_json_path
    vit_det_file_path = root_path + vit_det_file_path
    val_lvis_instances_results_vitdet = root_path + val_lvis_instances_results_vitdet

    # with open(vit_det_file_path) as f:
    #     res = json.load(f)
    # model_type = "vit_t"

    ######################################
    ########### json文件预处理 #############
    # all_files = os.listdir(val_img_path)
    # res_new = []
    # for i, res_ins in tqdm(enumerate(res)):
    #     res_ins = res[i]
    #     if i % 10000 == 0:
    #         print(i)
    #     img_id = res_ins['image_id']
    #     img_file_name = f'{img_id:012d}' + '.jpg'
    #     if img_file_name in all_files:
    #         res_new.append(res_ins)
    # with open('/home/ET/jtfan/MyData/Dataset/LVIS/json_files/val_lvis_instances_results_vitdet.json', 'w') as fnew:
    #     json.dump(res_new, fnew)
    ########### json文件预处理 #############
    ######################################

    with open(val_lvis_instances_results_vitdet) as f:
        res_new = json.load(f)

    pre_img_id = 0
    print("Total instances num:", len(res_new))
    Estimated_mIoU = 0
    mIoU = 0
    len_res = len(res_new)
    error = 0

    for i in tqdm(range(len(res_new))):

        res_ins = res_new[i]

        gt_mask = mask_util.decode(res_ins["segmentation"])
        img_id = res_ins['image_id']
        img_file_name = f'{img_id:012d}' + '.jpg'

        if model_name in ['efficient_sam_ti', 'efficient_sam_s']:
            input_box = res_ins['bbox']
            input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            input_point = np.reshape(np.array(input_box), (-1, 2))
            input_label = np.array([2, 3])

            if pre_img_id != img_id:
                image_np = np.array(Image.open(val_img_path + img_file_name))
                img_tensor = ToTensor()(image_np)
                predictor.set_image(img_tensor[None, ...])
                pre_img_id = img_id

            if image_np.shape[:-1] != gt_mask.shape:
                len_res -= 1
                error += 1
                continue

            masks, ious = run_ours_box_or_points(val_img_path + img_file_name,
                                                 input_point,
                                                 input_label,
                                                 predictor
                                                 )

            # predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            # show the masks #
            # show_mask(origin_image=image, mask=gt_mask)
            # show_mask(origin_image=image, mask=pred_mask)
            # print(calculate_iou(gt_mask, pred_mask))

            mIoU += calculate_iou(gt_mask, pred_mask)
            Estimated_mIoU += max(ious)

            # if i == 200:
            #     print(f"IoU(i==200) : {mIoU/200}")
            #     print("stop")
            new_seg = mask_util.encode(np.array(masks[np.argmax(ious)], order="F", dtype="uint8"))
            new_seg["counts"] = new_seg["counts"].decode("utf-8")
            res_ins["segmentation"] = new_seg


        else:
            if pre_img_id != img_id:
                image = cv2.imread(val_img_path + img_file_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                predictor.set_image(image)
                pre_img_id = img_id

            if image.shape[:-1] != gt_mask.shape:
                len_res -= 1
                error += 1
                continue

            input_box = res_ins['bbox']
            input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            input_box = np.array(input_box)

            t1_s = time.perf_counter()
            masks, ious, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
            )
            t1_e = time.perf_counter()
            t1 = t1_e - t1_s

            pred_mask = np.array(masks[np.argmax(ious)], order='F', dtype="uint8")

            # show the masks #
            # show_mask(origin_image=image, mask=gt_mask)
            # show_mask(origin_image=image, mask=pred_mask)
            # print(calculate_iou(gt_mask, pred_mask))

            mIoU += calculate_iou(gt_mask, pred_mask)
            Estimated_mIoU += max(ious)

            new_seg=mask_util.encode(np.array(masks[np.argmax(ious)],order="F", dtype="uint8"))
            new_seg["counts"] = new_seg["counts"].decode("utf-8")
            res_ins["segmentation"]=new_seg


    print(f"Finally total instances num: {len_res} , because there are {error} destroyed gt_masks")
    print(f"Estimated_mIoU : {Estimated_mIoU / len_res}")
    print(f"mIoU : {mIoU / len_res}")
    results = {
        'len_res': len_res,
        'error': error,
        'Estimated_mIoU': Estimated_mIoU / len_res,
        'mIoU': mIoU / len_res
    }

    # for c in res_new:
    #     c.pop("bbox", None)
    #
    #
    # save_res_json_file = root_path + 'json_files/' + eval_type + '_res_tinysam.json'
    #
    # with open(save_res_json_file, 'w') as fnew:
    #     json.dump(res_new, fnew)
    #
    # lvis_eval = LVISEval(val_json_path, save_res_json_file, "segm")
    # lvis_eval.run()
    # lvis_eval.print_results()
    #
    # os.remove(save_res_json_file)

    return results


def eval_lvis_box_multi(model_name, predictor, eval_type, root_path, val_img_path, val_json_path, vit_det_file_path,
                  val_lvis_instances_results_vitdet, num_cores):

    print("============== Evaluating on LVIS dataset:")

    val_img_path = root_path + val_img_path
    val_json_path = root_path + val_json_path
    vit_det_file_path = root_path + vit_det_file_path
    val_lvis_instances_results_vitdet = root_path + val_lvis_instances_results_vitdet

    with open(vit_det_file_path) as f:
        res = json.load(f)

    ######################################
    ########### json文件预处理 #############
    # all_files = os.listdir(val_img_path)
    # res_new = []
    # for i, res_ins in tqdm(enumerate(res)):
    #     res_ins = res[i]
    #     if i % 10000 == 0:
    #         print(i)
    #     img_id = res_ins['image_id']
    #     img_file_name = f'{img_id:012d}' + '.jpg'
    #     if img_file_name in all_files:
    #         res_new.append(res_ins)
    # with open('/home/ET/jtfan/MyData/Dataset/LVIS/json_files/val_lvis_instances_results_vitdet.json', 'w') as fnew:
    #     json.dump(res_new, fnew)
    ########### json文件预处理 #############
    ######################################

    with open(val_lvis_instances_results_vitdet) as f:
        res_new = json.load(f)

    res_multi_dict = {}
    for i in tqdm(range(len(res_new))):
        res_ins = res_new[i]
        img_id = res_ins['image_id']
        if img_id not in res_multi_dict:
            res_multi_dict[img_id] = [res_ins]
        else:
            res_multi_dict[img_id].append(res_ins)

    eval_input = [(model_name, res_i_list, predictor, img_id, val_img_path) for img_id, res_i_list in
                  res_multi_dict.items()]

    with multiprocessing.Pool(processes=num_cores) as pool:
        sum_miou_list = list(tqdm(pool.imap(eval_lvis_box_perimage, eval_input), total=len(eval_input), desc='evaling'))

    sum_miou_list = np.array(sum_miou_list)

    print(np.sum(sum_miou_list[:,0]) / (len(res_new)-np.sum(sum_miou_list[:,1])))


    print(f"Finally total instances num: {len(res_new)}")
    print(f"mIoU : {np.sum(sum_miou_list) / len(res_new)}")
    results = {
        'len_res': len(res_new),
        'mIoU': np.sum(sum_miou_list) / len(res_new)
    }

    # for c in res_new:
    #     c.pop("bbox", None)
    #
    # save_res_json_file = root_path + 'json_files/' + eval_type + '_res_tinysam.json'
    #
    # with open(save_res_json_file, 'w') as fnew:
    #     json.dump(res_new, fnew)
    #
    # lvis_eval = LVISEval(val_json_path, save_res_json_file, "segm")
    # lvis_eval.run()
    # lvis_eval.print_results()
    #
    # os.remove(save_res_json_file)

    return results


def eval_coco_point(model_name, predictor, root_path,  val_img_path, vit_det_file_path, point_from, num_points:Optional=1):

    val_img_path = root_path + val_img_path
    vit_det_file_path = root_path + vit_det_file_path

    with open(vit_det_file_path) as f:
        res = json.load(f)

    pre_img_id = 0
    print("Total instances num:", len(res))

    Estimated_mIoU = 0
    mIoU = 0
    len_res = len(res)
    error = 0
    for i in tqdm(range(len(res))):
        res_ins = res[i]

        img_id = res_ins['image_id']
        img_file_name = f'{img_id:012d}' + '.jpg'

        gt_mask = mask_util.decode(res[i]["segmentation"])

        input_box = res_ins['bbox']
        input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
        input_box = np.array(input_box)


        if model_name in ['efficient_sam_ti', 'efficient_sam_s']:
            input_box = res_ins['bbox']
            input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            input_point = np.reshape(np.array(input_box), (-1, 2))
            input_label = np.array([2, 3])

            if pre_img_id != img_id:
                image_np = np.array(Image.open(val_img_path + img_file_name))
                img_tensor = ToTensor()(image_np)
                predictor.set_image(img_tensor[None, ...])
                pre_img_id = img_id

            if image_np.shape[:-1] != gt_mask.shape or torch.tensor(gt_mask).nonzero().numel() == 0:
                len_res -= 1
                continue

            if point_from == 'mask-rand':
                prompt_points = get_points(point_from, gt_mask, input_box, num_points=num_points)
            else:
                prompt_points = get_points(point_from, gt_mask, input_box)

            masks, ious = run_ours_box_or_points(val_img_path + img_file_name,
                                                 prompt_points[0],
                                                 prompt_points[1],
                                                 predictor
                                                 )

            # mask_label & predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            # show the masks #
            # show_mask(origin_image=image_np, mask=seg_label)
            # show_mask(origin_image=image_np, mask=pred_mask, points_prompt=prompt_points[0])
            # print(calculate_iou(seg_label, pred_mask))

            mIoU += calculate_iou(gt_mask, pred_mask)
            Estimated_mIoU += max(ious)

            # if i == 200:
            #     print(f"mIoU(i==200) : {mIoU / 200}")
            #     print("stop")
            new_seg = mask_util.encode(np.array(masks[np.argmax(ious)], order="F", dtype="uint8"))
            new_seg["counts"] = new_seg["counts"].decode("utf-8")
            res[i]["segmentation"] = new_seg

        else:
            if pre_img_id != img_id:
                image = cv2.imread(val_img_path + img_file_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # plt.figure(figsize=(10, 10))
                # plt.imshow(image)
                # plt.axis('off')
                # plt.title("Original Image")
                # plt.show()
                predictor.set_image(image)
                pre_img_id = img_id

            if image.shape[:-1] != gt_mask.shape or torch.tensor(gt_mask).nonzero().numel() == 0:
                len_res -= 1
                error += 1
                # print(f"error : {error}")
                continue

            if point_from == 'mask-rand':
                prompt_points = get_points(point_from, gt_mask, input_box, num_points=num_points)
            else:
                prompt_points = get_points(point_from, gt_mask, input_box)

            masks, ious, _ = predictor.predict(
                point_coords=prompt_points[0],
                point_labels=prompt_points[1],
                box=None,
            )

            # mask_label & predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            # # show the masks #
            # show_mask(origin_image=image, mask=gt_mask)
            # show_mask(origin_image=image, mask=pred_mask, points_prompt=prompt_points[0])
            print(calculate_iou(gt_mask, pred_mask))

            mIoU += calculate_iou(gt_mask, pred_mask)
            Estimated_mIoU += max(ious)

            # if i == 200:
            #     print(f"mIoU(i==200) : {mIoU / 200}")
            #     print("stop")

            # new_seg = mask_util.encode(np.array(masks[np.argmax(ious)], order="F", dtype="uint8"))
            # new_seg["counts"] = new_seg["counts"].decode("utf-8")
            # res[i]["segmentation"] = new_seg

    print(f"Estimated_mIoU : {Estimated_mIoU / len_res}")
    print(f"mIoU : {mIoU / len_res}")
    print(f"Finally total instances num: {len_res} , because there are {error} destroyed gt_masks")
    results = {
        'len_res' : len_res,
        'error': error,
        'Estimated_mIoU': Estimated_mIoU / len_res,
        'mIoU': mIoU / len_res
    }

    return results


def eval_lvis_point(model_name, predictor, root_path, val_img_path, val_json_path, vit_det_file_path, val_lvis_instances_results_vitdet, point_from, num_points:Optional=1):
    print("============== Evaluating on LVIS dataset:")

    val_img_path = root_path + val_img_path
    val_json_path = root_path + val_json_path
    vit_det_file_path = root_path + vit_det_file_path
    val_lvis_instances_results_vitdet = root_path + val_lvis_instances_results_vitdet

    with open(vit_det_file_path) as f:
        res = json.load(f)

    ######################################
    ########### json文件预处理 #############
    # all_files = os.listdir(val_img_path)
    # res_new = []
    # for i, res_ins in tqdm(enumerate(res)):
    #     res_ins = res[i]
    #     if i % 10000 == 0:
    #         print(i)
    #     img_id = res_ins['image_id']
    #     img_file_name = f'{img_id:012d}' + '.jpg'
    #     if img_file_name in all_files:
    #         res_new.append(res_ins)
    # with open('/home/ET/jtfan/MyData/Dataset/LVIS/json_files/val_lvis_instances_results_vitdet.json', 'w') as fnew:
    #     json.dump(res_new, fnew)
    ########### json文件预处理 #############
    ######################################

    with open(val_lvis_instances_results_vitdet) as f:
        res_new = json.load(f)

    pre_img_id = 0
    print("Total instances num:", len(res_new))

    Estimated_mIoU = 0
    mIoU = 0
    len_res = len(res_new)
    error = 0
    for i in tqdm(range(len(res_new))):
        res_ins = res_new[i]

        img_id = res_ins['image_id']
        img_file_name = f'{img_id:012d}' + '.jpg'

        gt_mask = mask_util.decode(res_new[i]["segmentation"])

        input_box = res_ins['bbox']
        input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
        input_box = np.array(input_box)


        if model_name in ['efficient_sam_ti', 'efficient_sam_s']:
            input_box = res_ins['bbox']
            input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            input_point = np.reshape(np.array(input_box), (-1, 2))
            input_label = np.array([2, 3])

            if pre_img_id != img_id:
                image_np = np.array(Image.open(val_img_path + img_file_name))
                img_tensor = ToTensor()(image_np)
                predictor.set_image(img_tensor[None, ...])
                pre_img_id = img_id

            if image_np.shape[:-1] != gt_mask.shape or torch.tensor(gt_mask).nonzero().numel() == 0:
                len_res -= 1
                continue

            if point_from == 'mask-rand':
                prompt_points = get_points(point_from, gt_mask, input_box, num_points=num_points)
            else:
                prompt_points = get_points(point_from, gt_mask, input_box)

            masks, ious = run_ours_box_or_points(val_img_path + img_file_name,
                                                 prompt_points[0],
                                                 prompt_points[1],
                                                 predictor
                                                 )

            # mask_label & predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            # show the masks #
            # show_mask(origin_image=image_np, mask=seg_label)
            # show_mask(origin_image=image_np, mask=pred_mask, points_prompt=prompt_points[0])
            # print(calculate_iou(seg_label, pred_mask))

            mIoU += calculate_iou(gt_mask, pred_mask)
            Estimated_mIoU += max(ious)

            # if i == 200:
            #     print(f"mIoU(i==200) : {mIoU / 200}")
            #     print("stop")
            new_seg = mask_util.encode(np.array(masks[np.argmax(ious)], order="F", dtype="uint8"))
            new_seg["counts"] = new_seg["counts"].decode("utf-8")
            res_new[i]["segmentation"] = new_seg

        else:
            if pre_img_id != img_id:
                image = cv2.imread(val_img_path + img_file_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                predictor.set_image(image)
                pre_img_id = img_id

            if image.shape[:-1] != gt_mask.shape or torch.tensor(gt_mask).nonzero().numel() == 0:
                len_res -= 1
                error += 1
                # print(f"error : {error}")
                continue

            t2_s = time.perf_counter()
            if point_from == 'mask-rand':
                prompt_points = get_points(point_from, gt_mask, input_box, num_points=num_points)
            else:
                prompt_points = get_points(point_from, gt_mask, input_box)
            t2_e = time.perf_counter()
            t2 = t2_e - t2_s

            t1_s = time.perf_counter()
            masks, ious, _ = predictor.predict(
                point_coords=prompt_points[0],
                point_labels=prompt_points[1],
                box=None,
            )
            t1_e = time.perf_counter()
            t1 = t1_e - t1_s

            # mask_label & predict_mask
            pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

            # # show the masks #
            # show_mask(origin_image=image, mask=gt_mask)
            # show_mask(origin_image=image, mask=pred_mask, points_prompt=prompt_points[0])
            # print(calculate_iou(gt_mask, pred_mask))

            mIoU += calculate_iou(gt_mask, pred_mask)
            Estimated_mIoU += max(ious)

            # new_seg = mask_util.encode(np.array(masks[np.argmax(ious)], order="F", dtype="uint8"))
            # new_seg["counts"] = new_seg["counts"].decode("utf-8")
            # res[i]["segmentation"] = new_seg

    print(f"Estimated_mIoU : {Estimated_mIoU / len_res}")
    print(f"mIoU : {mIoU / len_res}")
    print(f"Finally total instances num: {len_res} , because there are {error} destroyed gt_masks")
    results = {
        'len_res' : len_res,
        'error': error,
        'Estimated_mIoU': Estimated_mIoU / len_res,
        'mIoU': mIoU / len_res
    }

    return results


def get_prompt(item_dict, device, point_from, num_points, model_name): # (index, item_dict)

    index = item_dict[0]
    item_dict = item_dict[1]
    input_box = item_dict['bbox']
    gt_mask = mask_util.decode(item_dict["segmentation"])


    try:
        if model_name in ['efficient_sam_ti', 'efficient_sam_s']:
            input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
        else:
            input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
            input_box = np.array(input_box)

        prompt = get_points(point_from=point_from, gt_mask=gt_mask, boxes=input_box, device=device, num_points=num_points)
        return [index, prompt, True]

    except:
        return [index, None, False]


    # if torch.tensor(gt_mask).nonzero().numel() == 0:
    #     return [index, None, False]
    #
    # else:
    #     if model_name in ['efficient_sam_ti', 'efficient_sam_s']:
    #         input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
    #     else:
    #         input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
    #         input_box = np.array(input_box)
    #
    #     prompt = get_points(point_from=point_from, gt_mask=gt_mask, boxes=input_box, device=device, num_points=num_points)
    #     return [index, prompt, True]


def get_box_center(item_dict): # (index, item_dict)
    index = item_dict[0]
    item_dict = item_dict[1]

    input_box = item_dict['bbox']
    gt_mask = mask_util.decode(item_dict["segmentation"])
    input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
    prompt = get_points('box-center', gt_mask, input_box)

    return [index, prompt]

def get_mask_rand_1(item_dict): # (index, item_dict)
    index = item_dict[0]
    item_dict = item_dict[1]

    input_box = item_dict['bbox']
    gt_mask = mask_util.decode(item_dict["segmentation"])
    input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
    prompt = get_points('mask-rand', gt_mask, input_box, num_points=1)

    return [index, prompt]


def get_mask_rand_2(item_dict): # (index, item_dict)
    index = item_dict[0]
    item_dict = item_dict[1]

    input_box = item_dict['bbox']
    gt_mask = mask_util.decode(item_dict["segmentation"])
    input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
    prompt = get_points('mask-rand', gt_mask, input_box, num_points=2)

    return [index, prompt]


def eval_lvis_point_multi(model_name, predictor, root_path, val_img_path, val_json_path, vit_det_file_path, val_lvis_instances_results_vitdet, point_from, num_cores, device, num_points:Optional=1):
    print("============== Evaluating on LVIS dataset:")

    val_img_path = root_path + val_img_path
    val_json_path = root_path + val_json_path
    vit_det_file_path = root_path + vit_det_file_path
    val_lvis_instances_results_vitdet = root_path + val_lvis_instances_results_vitdet

    with open(vit_det_file_path) as f:
        res = json.load(f)

    #####################################
    ########## json文件预处理 #############
    if os.path.exists(val_lvis_instances_results_vitdet):
        with open(val_lvis_instances_results_vitdet) as f:
            res_new = json.load(f)
    else:
        all_files = os.listdir(val_img_path)
        res_new = []
        for i, res_ins in tqdm(enumerate(res)):
            res_ins = res[i]
            img_id = res_ins['image_id']
            img_file_name = f'{img_id:012d}' + '.jpg'
            if img_file_name in all_files:
                res_new.append(res_ins)
        with open(val_lvis_instances_results_vitdet, 'w') as fnew:
            json.dump(res_new, fnew)
    ########## json文件预处理 #############
    #####################################

    if point_from != "mask-rand":
        prompt_file_name = root_path + f'json_files/lvis_{point_from}.pkl'
    else:
        prompt_file_name = root_path + f'json_files/lvis_{point_from}_{str(num_points)}.pkl'

    if os.path.exists(prompt_file_name):
        with open(prompt_file_name, 'rb') as f:
            res_new = pickle.load(f)
    else:
        get_prompts_num_cores = 80 #80

        from functools import partial
        partial_add = partial(get_prompt, device=device, point_from=point_from, num_points=num_points, model_name=model_name)

        with multiprocessing.Pool(processes=get_prompts_num_cores) as pool:
            prompts = list(tqdm(pool.imap(partial_add, enumerate(res_new)), total=len(res_new),
                                desc='Get Prompts ...'))

        for i in prompts:
            if i[2]:
                res_new[i[0]]['prompt'] = i[1]
            else:
                res_new[i[0]]['prompt'] = None

        with open(prompt_file_name, 'wb') as fnew:
            pickle.dump(res_new, fnew)
        torch.cuda.empty_cache()


    res_multi_dict = {}
    for i in tqdm(range(len(res_new))):
        res_ins = res_new[i]
        img_id = res_ins['image_id']

        if img_id not in res_multi_dict:
            res_multi_dict[img_id] = [res_ins]
        else:
            res_multi_dict[img_id].append(res_ins)


    eval_input = [(model_name, res_i_list, predictor , img_id, val_img_path, point_from, num_points) for img_id, res_i_list in
                  res_multi_dict.items()]

    with multiprocessing.Pool(processes=num_cores) as pool:
        sum_miou_list = list(tqdm(pool.imap(eval_lvis_point_perimage, eval_input), total=len(eval_input), desc='evaling'))

    sum_miou_list = np.array(sum_miou_list)

    miou = np.sum(sum_miou_list[:, 0]) / np.sum(sum_miou_list[:, 1])
    print(miou)

    print(f"Finally total instances num: {np.sum(sum_miou_list[:, 1])}")
    print(f"mIoU : {miou}")
    results = {
        'len_res': np.sum(sum_miou_list[:, 1]),
        'mIoU': miou
    }

    # for c in res_new:
    #     c.pop("bbox", None)
    #
    # save_res_json_file = root_path + 'json_files/' + eval_type + '_res_tinysam.json'
    #
    # with open(save_res_json_file, 'w') as fnew:
    #     json.dump(res_new, fnew)
    #
    # lvis_eval = LVISEval(val_json_path, save_res_json_file, "segm")
    # lvis_eval.run()
    # lvis_eval.print_results()
    #
    # os.remove(save_res_json_file)

    return results
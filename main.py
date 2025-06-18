import torch
import os
import shutil
import time
from sam_family.load_model import SegAny
from utils.model_encode import compress_params
from utils.model_decode import decompress_params
from utils.model_hyperload import load_from_hyperencoded
from safetensors.torch import save_file, load_file
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
from torchvision.transforms import ToTensor
from typing import Optional
from tqdm import tqdm
import json
import copy
import random


def random_color():
    return (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

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


    return points

def seg_box(model_class, img_file_name, res_ins):

    res_ins_new = copy.deepcopy(res_ins)
    predictor = model_class.predictor_with_point_prompt

    image_np = np.array(Image.open(img_file_name))
    image_tensor = ToTensor()(image_np)
    image = image_tensor[None, ...]

    image = cv2.imread(img_file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    input_box = res_ins['bbox']
    input_box = [input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3]]
    input_box = np.array(input_box)

    gt_mask = mask_util.decode(res_ins_new["segmentation"])
    prompt_points = get_points('mask-rand', gt_mask, input_box, num_points=3)


    masks, ious, _ = predictor.predict(
        point_coords=prompt_points[0],
        point_labels=prompt_points[1],
        box=input_box[None, :],
    )

    # predict_mask
    pred_mask = np.array(masks[np.argmax(ious)], order="F", dtype="uint8")

    new_seg = mask_util.encode(np.array(masks[np.argmax(ious)], order="F", dtype="uint8"))
    res_ins_new["counts"] = new_seg["counts"].decode("utf-8")
    res_ins_new["segmentation"] = new_seg

    return res_ins_new



if __name__ == '__main__':

    rect_l = 0.1
    num_inner_list = [1225, 1600]
    class_max = 3
    loss_max = 0.001
    loss_hope = 0.001
    stop_threshold = [True, 0.006]


    mode = 'inference'  # 'encode' or 'inference'
    model_name = 'sam_hq_vit_h'
    device = 'cuda:3'

    # encoding save path
    Save_Param_Path = f'./compressed_result/{model_name}/'


    check_points = './sam_family/checkpoints/sam_hq_vit_h.pth'
    # check_points = '/home/ET/jtfan/MyData/checkpoints/sam_hq_vit_h.pth'

    if mode == 'encode':
        model_class = SegAny(model_name=model_name,
                             checkpoint=check_points,
                             device=device,
                             hyper_load=False)
        model_predictor = model_class.predictor_with_point_prompt
        model = model_class.model


        if os.path.exists(Save_Param_Path):
            shutil.rmtree(Save_Param_Path)
            os.makedirs(Save_Param_Path, exist_ok=True)
        else:
            os.makedirs(Save_Param_Path, exist_ok=True)


        t_start = time.perf_counter()

        encoded_dict, back_dict = compress_params(model, rect_l, num_inner_list, class_max,
                                             loss_max, loss_hope, stop_threshold, device)
        t_end = time.perf_counter()

        save_file(encoded_dict, Save_Param_Path+'hyper_model.safetensors')
        save_file(back_dict, Save_Param_Path + 'back_model.safetensors')

        print(f"Size of Original Checkpoints : {os.path.getsize(check_points)} Bytes")
        print(f"Size of Compressed Checkpoints : {os.path.getsize(Save_Param_Path+'hyper_model.safetensors')} Bytes")
        print(f"Ratio : {os.path.getsize(check_points) / os.path.getsize(Save_Param_Path+'hyper_model.safetensors')}")

        print("Encoding Finished!!!!!!!")
        print(f"Compression Time : {t_end - t_start} Seconds")
        print("Done!")


    elif mode == 'inference':

        index_cuda = 0
        device_hyper = f'cuda:{index_cuda}'
        device = f'cuda:{index_cuda}'

        hyper_file = Save_Param_Path + 'hyper_model.safetensors'
        hyper_model_class = SegAny(model_name=model_name,
                             checkpoint=None,
                             device=device_hyper,
                             hyper_load=True,
                             hyper_file=hyper_file
                             )
        hyper_model_predictor = hyper_model_class.predictor_with_point_prompt
        hyper_model = hyper_model_class.model



        model_class = SegAny(model_name=model_name,
                                   checkpoint=check_points,
                                   device=device,
                                   hyper_load=False
                                   )
        model_predictor = model_class.predictor_with_point_prompt
        model = model_class.model


        ###########################
        # Segment Everything Task #
        ###########################
        img_path = './test_data/sa_145444.jpg'
        ann_path = './test_data/sa_145444.json'

        img = cv2.imread(img_path)
        img_hyper = cv2.imread(img_path)
        ann = json.load(open(ann_path))

        mask_hyper = np.zeros_like(img, dtype=np.uint8)
        mask = np.zeros_like(img, dtype=np.uint8)
        for obj in tqdm(ann['annotations']):
            color = random_color()

            obj_hyper = seg_box(hyper_model_class, img_path, obj)
            seg_hyper = mask_util.decode(obj_hyper['segmentation'])
            mask_hyper[seg_hyper > 0] = color


        if mask_hyper.ndim == 2:
            mask_hyper = np.stack([mask_hyper] * 3, axis=-1)
        blended = (0.5 * img + 0.5 * mask_hyper).astype(np.uint8)
        Image.fromarray(blended).save(f'./test_data/seg_results/seg_every_hyper.png')






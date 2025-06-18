from tqdm import tqdm
import torch
from torch import nn
from safetensors.torch import load_file
import time
import matplotlib.pyplot as plt
from utils.model_encode import get_Try_torch, decompression_1d_torch
import gc

def decompress_params(layer_names, file, decode_mode, device):
    """
        layer_names : the names of every tensors in model.state_dict()
        file : the path of encoded params
    """

    encoded_dict = load_file(file)
    decoded_dict = {}
    for name in layer_names:

        # print(name)

        if encoded_dict[name+'.load_type'] == 0:
            decoded_dict[name] = encoded_dict[name+'.origin_param'].to('cpu')
            del encoded_dict[name+'.origin_param']
        else:
            aux = {
                'rect_l' : encoded_dict['rect_l'].to(device),
                'num_inner_list' : encoded_dict['num_inner_list'].to(device),
                'if_padding' : encoded_dict[name + '.if_padding'].to(device),
                'center_node' : encoded_dict[name + '.center_node'].to(device),
                'farthest_node' : encoded_dict[name + '.farthest_node'].to(device),
                'U' : encoded_dict[name + '.U'].to(device),
                'K' : encoded_dict[name + '.K'].to(device),
                'original_shape' : tuple(encoded_dict[name + '.original_shape'].tolist()),
                'padding_bits' : encoded_dict[name + '.padding_bits'].to(device),
                'uint_i' : encoded_dict[name + '.uint_i'].to(device),
            }

            U = aux['num_inner_list'][aux['U'].item()].to(dtype=torch.float32)
            rect_l = aux['rect_l'].to(dtype=torch.float32)
            center = aux['center_node']
            farthest = aux['farthest_node']
            K = aux['K']
            farthest_dis = torch.linalg.norm(center - farthest).to(torch.float32)
            each_dis = float((farthest_dis - (rect_l / 2)) / K)
            sqrt_n_ceil = torch.ceil(torch.sqrt(U))
            d_inner = rect_l / sqrt_n_ceil  # inner矩阵的d
            l1 = torch.sqrt(rect_l ** 2 + d_inner ** 2)
            tan_alpha = sqrt_n_ceil
            sin_alpha = rect_l / l1
            step_size = (rect_l / sin_alpha) / sqrt_n_ceil
            node_ld = center - torch.tensor([rect_l / 2, rect_l / 2], dtype=torch.float32).to(device)
            a = torch.tensor([rect_l / tan_alpha, rect_l]).to(device)
            a = a / torch.linalg.norm(a)
            a_jump = a * step_size

            encoded_index = encoded_dict[name + '.encoded_index'].to(device)
            encoded_index = restore_from_uint8_tensor(encoded_index, aux['uint_i'], aux['padding_bits'])

            if encoded_dict[name+'.load_type'] == 1:
                decoded_param = decode_param_type_1(encoded_index, aux, device)
                decoded_dict[name] = decoded_param.to('cpu')

            elif encoded_dict[name+'.load_type'] == 2:
                decoded_param = decode_param_type_2(encoded_index, aux, device, decode_mode).to('cpu')

                if decode_mode == 'standard':
                    decoded_dict[name] = decoded_param.to('cpu')

                elif decode_mode == 'hyper_op' and '.weight' in name:
                    decoded_dict[name[:-6] + 'index_matrix'] = decoded_param.detach().clone().detach().to('cpu')
                    decoded_dict[name[:-6] + 'each_dis'] = torch.tensor([each_dis], dtype=torch.float32).clone().detach().to('cpu')
                    decoded_dict[name[:-6] + 'U'] = torch.tensor([U.int()]).detach().clone().detach().to('cpu')
                    decoded_dict[name[:-6] + 'rect_l'] = torch.tensor([rect_l], dtype=torch.float16).clone().detach().to('cpu')
                    decoded_dict[name[:-6] + 'K'] = torch.tensor([K], dtype=torch.uint8).clone().detach().to('cpu')
                    decoded_dict[name[:-6] + 'node_ld'] = node_ld.clone().detach().to('cpu')
                    decoded_dict[name[:-6] + 'a_jump'] = a_jump.clone().detach().to('cpu')
                    decoded_dict[name[:-6] + 'center'] = center.clone().detach().to('cpu')
                else:
                    raise ValueError(f'{name} cannot be loaded as an index_matrix.')

            del decoded_param
            del encoded_dict[name + '.if_padding']
            del encoded_dict[name + '.center_node']
            del encoded_dict[name + '.farthest_node']
            del encoded_dict[name + '.U']
            del encoded_dict[name + '.K']
            del encoded_dict[name + '.original_shape']
            del encoded_dict[name + '.padding_bits']
            del encoded_dict[name + '.uint_i']

        decoded_dict[name + '.load_type'] = encoded_dict[name + '.load_type'].to('cpu')
        del encoded_dict[name + '.load_type']


    print(torch.cuda.memory_allocated(device))

    del encoded_dict

    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated(device))

    gc.collect()
    torch.cuda.empty_cache()

    return decoded_dict



def decode_param_type_1(encoded_index, aux, device):
    U = aux['num_inner_list'][aux['U'].item()].to(dtype=torch.float32).to(device)
    rect_l = aux['rect_l'].to(device)
    center = aux['center_node'].to(device)
    farthest = aux['farthest_node'].to(device)
    K = aux['K'].to(device)
    original_shape = aux['original_shape']
    if_padding = aux['if_padding'].to(device)
    sqrt_n_ceil = torch.ceil(torch.sqrt(U))
    d_inner = rect_l / sqrt_n_ceil  # inner矩阵的d
    l1 = torch.sqrt(rect_l ** 2 + d_inner ** 2)
    tan_alpha = sqrt_n_ceil
    sin_alpha = rect_l / l1
    step_size = (rect_l / sin_alpha) / sqrt_n_ceil


    Try_rect_inner = get_Try_torch(tan_alpha,
                                   step_size=step_size,
                                   U=U,
                                   node_ld=(center - torch.tensor([rect_l / 2, rect_l / 2]).to(center.device)),
                                   rect_l=rect_l
                                   ).reshape([-1, 2])

    if K != 0:  # K != 0
        index_matrix = encoded_index.to(dtype=torch.int64)
        farthest_dis = torch.linalg.norm(center - farthest)
        index_class = torch.floor(index_matrix / U)  # C
        each_dis = (farthest_dis - (rect_l / 2)) / K
        dis_list = rect_l / 2 + index_class * each_dis
        factor = ((rect_l / 2) / dis_list).reshape(-1, 1)

        index_matrix = index_matrix - index_class * U
        B_star = decompression_1d_torch(index_matrix.reshape(-1, 1), Try_rect_inner)
        B_star = B_star.reshape(-1, 2)
        # center_node = torch.tile(torch.tensor([center]), (B_star.shape[0], 1))
        OB_star = B_star - center
        OC_star = OB_star / factor
        C_star = OC_star + center


    else:  # K == 0
        # index_array_i = index_array
        C_star = decompression_1d_torch(encoded_index.reshape(-1, 1), Try_rect_inner)

    decoded_tensor = C_star.flatten()
    if if_padding != 0:
        decoded_tensor = decoded_tensor[:-int(if_padding)]
    decoded_tensor = decoded_tensor.reshape(original_shape)

    return decoded_tensor


def decode_param_type_2(encoded_index, aux, device, decode_mode):
    M, N = aux['original_shape']

    if decode_mode == 'hyper_op':
        decoded_tensor = encoded_index.reshape(M, -1)

    elif decode_mode == 'standard':
        U = aux['num_inner_list'][aux['U'].item()].to(dtype=torch.float32).to(device)
        rect_l = aux['rect_l'].to(dtype=torch.float32).to(device)
        center = aux['center_node'].to(device)
        farthest = aux['farthest_node'].to(device)
        K = aux['K'].to(device)
        if_padding = aux['if_padding'].to(device)
        sqrt_n_ceil = torch.ceil(torch.sqrt(U))
        d_inner = rect_l / sqrt_n_ceil  # inner矩阵的d
        l1 = torch.sqrt(rect_l ** 2 + d_inner ** 2)
        tan_alpha = sqrt_n_ceil
        sin_alpha = rect_l / l1
        step_size = (rect_l / sin_alpha) / sqrt_n_ceil


        Try_rect_inner = get_Try_torch(tan_alpha,
                                       step_size=step_size,
                                       U=U,
                                       node_ld=center - torch.tensor([rect_l / 2, rect_l / 2]).to(center.device),
                                       rect_l=rect_l
                                       ).reshape([-1, 2])

        if K != 0:  # K != 0
            index_matrix = encoded_index.to(dtype=torch.int64)
            farthest_dis = torch.linalg.norm(center - farthest)
            index_class = torch.floor(index_matrix / U)  # C
            each_dis = (farthest_dis - (rect_l / 2)) / K
            dis_list = rect_l / 2 + index_class * each_dis
            factor = ((rect_l / 2) / dis_list).reshape(-1, 1)
            index_matrix = index_matrix - index_class * U

            B_star = decompression_1d_torch(index_matrix.reshape(-1, 1), Try_rect_inner)
            B_star = B_star.reshape(-1, 2)
            # center_node = torch.tile(torch.tensor([center]), (B_star.shape[0], 1))
            OB_star = B_star - center
            OC_star = OB_star / factor
            C_star = OC_star + center

        else:  # K == 0
            # index_array_i = index_array
            C_star = decompression_1d_torch(encoded_index.reshape(-1, 1), Try_rect_inner)

        decoded_tensor = C_star.flatten().reshape(M, -1)
        if if_padding != 0:
            # decoded_tensor = decoded_tensor[:, :-1].T
            decoded_tensor = decoded_tensor[:, :-1]
        else:
            # decoded_tensor = decoded_tensor.T
            decoded_tensor = decoded_tensor

    return decoded_tensor


def restore_from_uint8_tensor(uint8_arr, bits_per_int, pad):

    bits = torch.bitwise_right_shift(uint8_arr.unsqueeze(-1), torch.arange(7, -1, -1, device=uint8_arr.device)) & 1
    bits = bits.flatten()

    # 去除填充的零
    if pad > 0:
        bits = bits[pad:]

    bits = bits.view(-1, bits_per_int)

    shifts = torch.arange(bits_per_int - 1, -1, -1, device=bits.device)
    restored_arr = (bits << shifts).sum(dim=1)

    max_value = restored_arr.int().max().item()

    if max_value <= 255:
        dtype = torch.uint8  # 8 位无符号整数
    elif max_value <= 65535:
        dtype = torch.uint16  # 16 位无符号整数
    elif max_value <= 4294967295:
        dtype = torch.uint32  # 32 位无符号整数
    else:
        dtype = torch.uint64  # 64 位无符号整数

    # 将张量转换为目标 dtype
    restored_arr = restored_arr.to(dtype)

    return restored_arr
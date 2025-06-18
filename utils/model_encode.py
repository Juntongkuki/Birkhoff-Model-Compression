from tqdm import tqdm
import torch
from torch import nn
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def compress_params(model,
                    rect_l,
                    num_inner_list,
                    class_max,
                    loss_max,
                    loss_hope,
                    stop_threshold,
                    device
                    ):

    # Get all blocks
    blocks = []
    for name, module in model.named_children():
        blocks.append((name, module))

    # Get all not-buffers parameters names
    model_parameters_names = []
    for n, t in model.named_parameters():
        model_parameters_names.append(n)


    num_total = sum(p.numel() for p in model.state_dict().values())
    num_linear = 0
    linear_tensor_names = []  # Get all tensor_names of nn.linear()
    for i in tqdm(range(len(blocks))):
        name_block = blocks[i][0]
        block = blocks[i][1]
        for m_name, m in block.named_modules():
            if isinstance(m, nn.Linear):
                if m_name != '':
                    linear_tensor_names.append(name_block + '.' + m_name+'.weight')
                    num_linear += m.state_dict()['weight'].numel()
                    if m.bias is not None:
                        linear_tensor_names.append(name_block + '.' + m_name+'.bias')
                else:
                    linear_tensor_names.append(name_block + '.weight')
                    num_linear += m.state_dict()['weight'].numel()
                    if m.bias is not None:
                        linear_tensor_names.append(name_block + '.bias')
    print(f"The ratio of parameters from nn.linear() : {num_linear / num_total}")

    # test #
    for tensor_name in linear_tensor_names:
        if tensor_name not in model.state_dict():
            raise NotImplementedError("Some linear parameters lost.")


    # create params' ready2encode dict
    ready2encode = []
    encoded_dict = {}  # encoded_results --> safetensors
    for tensor_name, tensor in model.state_dict().items():
        if tensor_name == 'prompt_encoder.pe_layer.positional_encoding_gaussian_matrix':
            print("")

        if tensor_name not in model_parameters_names:  # This is a buffer, save directly
            encoded_dict[tensor_name + '.origin_param'] = tensor
            encoded_dict[tensor_name + '.load_type'] = torch.tensor([0], dtype=torch.uint8)
            continue

        elif tensor.numel() < 10:  # numel() < 10, save directly
            encoded_dict[tensor_name + '.origin_param'] = tensor
            encoded_dict[tensor_name + '.load_type'] = torch.tensor([0], dtype=torch.uint8)
            continue

        elif tensor_name in linear_tensor_names and ".weight" in tensor_name:
            ready2encode.append([tensor_name, tensor.to(device), 'linear'])  # 'linear' : it is a "nn.linear.weight"
            continue

        else:
            ready2encode.append([tensor_name, tensor.to(device), 'non-linear'])  # 'non-linear' : it is a "nn.linear.bias" or weights not from nn.linear()
            continue

    stop_threshold[1] = torch.tensor([stop_threshold[1]], dtype=torch.float32)
    aux = {
        'rect_l': torch.tensor([rect_l], dtype=torch.float32),
        'num_inner': torch.tensor(num_inner_list, dtype=torch.float32),
        'class_max': torch.tensor([class_max], dtype=torch.float32),
        'loss_max': torch.tensor([loss_max], dtype=torch.float32),
        'loss_hope': torch.tensor([loss_hope], dtype=torch.float32),
        'stop_threshold': stop_threshold
    }


    new_params_multi_list = []
    for i in range(len(ready2encode)):
        new_params_multi_list.append(encode_tensor_torch_version(ready2encode[i], aux, device, i))


    total_loss = 0
    loss_num = 0
    mae_loss_min = 100
    mae_loss_max = -100

    back_dict = encoded_dict.copy()
    encoded_dict['rect_l'] = torch.tensor(rect_l, dtype=torch.float32)
    encoded_dict['num_inner_list'] = torch.tensor(num_inner_list, dtype=torch.int64)
    for item in new_params_multi_list:
        tensor_name = item['tensor_name']
        load_type = item['load_type']

        if load_type == 0:
            encoded_dict[tensor_name+'.load_type'] = load_type.to('cpu')
            encoded_dict[tensor_name + '.origin_param'] = item['origin_param'].to('cpu')
            back_dict[tensor_name] = item['origin_param']

        elif load_type == 1:
            back_dict[tensor_name] = item['back_tensor'].contiguous()
            total_loss += item['mae']
            loss_num += 1

            if item['mae'] < mae_loss_min:
                mae_loss_min = item['mae']

            if item['mae'] > mae_loss_max:
                mae_loss_max = item['mae']

            encoded_dict[tensor_name + '.load_type'] = load_type.to('cpu')
            encoded_dict[tensor_name + '.encoded_index'] = item['encoded_index'].to('cpu')
            encoded_dict[tensor_name + '.if_padding'] = item['if_padding'].to('cpu')
            encoded_dict[tensor_name + '.center_node'] = item['center_node'].to('cpu')
            encoded_dict[tensor_name + '.farthest_node'] = item['farthest_node'].to('cpu')
            encoded_dict[tensor_name + '.U'] = item['U'].to('cpu')
            encoded_dict[tensor_name + '.K'] = item['K'].to('cpu')
            encoded_dict[tensor_name + '.original_shape'] = item['original_shape'].to('cpu')
            encoded_dict[tensor_name + '.padding_bits'] = item['padding_bits'].to('cpu')
            encoded_dict[tensor_name + '.uint_i'] = item['uint_i'].to('cpu')

        elif load_type == 2:
            back_dict[tensor_name] = item['back_tensor'].contiguous()
            total_loss += item['mae']
            loss_num += 1

            if item['mae'] < mae_loss_min:
                mae_loss_min = item['mae']

            if item['mae'] > mae_loss_max:
                mae_loss_max = item['mae']

            encoded_dict[tensor_name + '.load_type'] = load_type.to('cpu')
            encoded_dict[tensor_name + '.encoded_index'] = item['encoded_index'].to('cpu')
            encoded_dict[tensor_name + '.if_padding'] = item['if_padding'].to('cpu')
            encoded_dict[tensor_name + '.center_node'] = item['center_node'].to('cpu')
            encoded_dict[tensor_name + '.farthest_node'] = item['farthest_node'].to('cpu')
            encoded_dict[tensor_name + '.U'] = item['U'].to('cpu')
            encoded_dict[tensor_name + '.K'] = item['K'].to('cpu')
            encoded_dict[tensor_name + '.original_shape'] = item['original_shape'].to('cpu')
            encoded_dict[tensor_name + '.padding_bits'] = item['padding_bits'].to('cpu')
            encoded_dict[tensor_name + '.uint_i'] = item['uint_i'].to('cpu')


    return encoded_dict, back_dict




def encode_tensor_torch_version(tensor, aux, device, i):

    """
        tensor : [tensor_name, tensor, type] (type : ['linear', 'non-linear'])
        aux : the hyper_params needed
        device : the device used
    """

    print("\n")
    print(i)
    print(f"Name of tensor : {tensor[0]},    shape : {tensor[1].shape},   numel : {tensor[1].numel()}, type : {tensor[2]}")

    t1s = time.perf_counter()

    rect_l = aux['rect_l'].to(device)
    num_inner = aux['num_inner'].to(device)
    class_max = aux['class_max'].to(device)
    loss_max = aux['loss_max'].to(device)
    loss_hope = aux['loss_hope'].to(device)
    stop_threshold = aux['stop_threshold']
    stop_threshold[1] = stop_threshold[1].to(device)

    if tensor[2] == 'linear': # linear tensor
        # tensor[1] = tensor[1].T
        # tensor = (tensor[0], tensor[1].T, tensor[2])
        # M, N = tensor[1].shape
        original_shape = tensor[1].shape
        numel = tensor[1].numel()
        load_type = 2
        if original_shape[1] % 2 == 0:
            if_padding = 0  # no padding
            tensor_split = tensor[1].flatten().reshape(-1, 2)

        else:
            if_padding = 1  # need padding
            original_shape[1] += 1
            tensor_part = tensor[1][:, 0:-1].flatten().reshape(1, -1)
            tensor_part_point = tensor_part.reshape(-1, 2)
            padding = torch.mean(tensor_part_point[:, 1]).repeat(original_shape[0]).reshape(-1, 1)
            new_tensor = torch.cat((tensor[1], padding), dim=1)
            tensor_split = new_tensor.flatten().reshape(-1, 2)

    elif tensor[2] == 'non-linear': # non-linear
        # M, N = tensor[1].shape
        original_shape = tensor[1].shape
        numel = tensor[1].numel()
        load_type = 1
        if numel % 2 == 0:
            if_padding = 0  # no padding
            tensor_split = tensor[1].flatten().reshape(-1, 2)
        else:
            if_padding = 1  # need padding
            tensor_flat = tensor[1].flatten()[:-1]
            tensor_flat = tensor_flat.reshape(-1,2)
            padding = torch.mean(tensor_flat[:, 1]).unsqueeze(0)
            tensor_split = torch.cat((tensor[1].flatten(), padding)).reshape(-1, 2)

    t1e = time.perf_counter()
    t1 = t1e - t1s
    print(f"t1 : {t1}")

    """
    results[0] : MAE_loss list
    results[1] ：best categories
    results[2] ：new params
    results[3] : compressed index
    results[4] : number of inner nodes (except for center node)
    results[5] : number of inner nodes
    results[6] : number of outer nodes
    results[7] : padding_size
    results[8] : num_inner's index in num_inner_list
    results[9] : center_node
    results[10] : farthest_node
    """
    encode_result = {}
    results = compress_decom_v3(tensor_split, tensor[0], rect_l, num_inner, class_max, loss_max, loss_hope, device)
    mean_MAE = torch.mean(results[0]).item()


    t5s = time.perf_counter()
    if stop_threshold[0] and mean_MAE > stop_threshold[1]:  # 启用stop_threshold 且 mae > threshold， 则直接保存
        print(f"mean_MAE > stop_threshold, numel = {results[2].numel()}")
        load_type = 0
        encode_result['load_type'] = torch.tensor(load_type).to(dtype=torch.uint8)
        encode_result['tensor_name'] = tensor[0]
        if tensor[2] == 'linear':
            encode_result['origin_param'] = tensor[1]
        else:
            encode_result['origin_param'] = tensor[1]


    elif stop_threshold[0] and mean_MAE <= stop_threshold[1]:  # 启用stop_threshold 且 mae <= threshold, 则正常压缩并保存


        encode_result['mae'] = mean_MAE

        if tensor[2] == 'linear':
            if if_padding == 1:
                # encode_result['back_tensor'] = results[2].reshape(M, -1)[:, :-1].T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1)[:, :-1].to("cpu")
            else:
                # encode_result['back_tensor'] = results[2].reshape(M, -1).T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1).to("cpu")
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(results[3].flatten())

        elif tensor[2] == 'non-linear':
            if if_padding == 1:
                encode_result['back_tensor'] = results[2].flatten()[:-1].reshape(original_shape).to("cpu")
            else:
                encode_result['back_tensor'] = results[2].flatten().reshape(original_shape).to("cpu")
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(results[3])

        encode_result['load_type'] = torch.tensor(load_type).to(dtype=torch.uint8)
        encode_result['if_padding'] = torch.tensor(if_padding).to(dtype=torch.uint8)
        encode_result['center_node'] = results[9].to(dtype=torch.float32)
        encode_result['farthest_node'] = results[10].to(dtype=torch.float32)
        encode_result['U'] = torch.where(results[4].item() == num_inner)[0][0].to(dtype=torch.uint8)
        encode_result['K'] = torch.tensor(results[1]).to(dtype=torch.uint8)
        encode_result['tensor_name'] = tensor[0]
        encode_result['original_shape'] = torch.tensor(original_shape).to(dtype=torch.int64)

        new_param = results[2]
        max_loss = torch.max(results[0])
        min_loss = torch.min(results[0])
        max_index = (results[4] + 1) + results[1] * (results[4] + 1)
        if_padding = results[7]
        num_inner_index = results[8]
        best_class = results[1]
        center_node = results[9]
        farthest_node = results[10]
        # print(f"total_mean_loss : {mean_MAE}")
        # print(f"best_class : {results[1]}")
        # print(f"max_index : {max_index}")
        # print(f"real_max_index : {torch.max(results[3].int())}")
        # print(f"num_inside : {results[5]}")
        # print(f"num_outside : {results[6]}")

    elif not stop_threshold[0]:

        encode_result['mae'] = mean_MAE

        if tensor[2] == 'linear':
            if if_padding == 1:
                # encode_result['back_tensor'] = results[2].reshape(M, -1)[:, :-1].T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1)[:, :-1].to("cpu")
            else:
                # encode_result['back_tensor'] = results[2].reshape(M, -1).T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1).to("cpu")
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(
                results[3].flatten())

        elif tensor[2] == 'non-linear':
            if if_padding == 1:
                encode_result['back_tensor'] = results[2].flatten()[:-1].reshape(original_shape).to("cpu")
            else:
                encode_result['back_tensor'] = results[2].flatten().reshape(original_shape).to("cpu")
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(results[3])

        encode_result['load_type'] = torch.tensor(load_type).to(dtype=torch.uint8)
        encode_result['if_padding'] = torch.tensor(if_padding).to(dtype=torch.uint8)
        encode_result['center_node'] = results[9].to(dtype=torch.float32)
        encode_result['farthest_node'] = results[10].to(dtype=torch.float32)
        encode_result['U'] = torch.where(results[4].item() == num_inner)[0][0].to(dtype=torch.uint8)
        encode_result['K'] = torch.tensor(results[1]).to(dtype=torch.uint8)
        encode_result['tensor_name'] = tensor[0]
        encode_result['original_shape'] = torch.tensor(original_shape).to(dtype=torch.int64)

        new_param = results[2]
        max_loss = torch.max(results[0])
        min_loss = torch.min(results[0])
        max_index = (results[4] + 1) + results[1] * (results[4] + 1)
        if_padding = results[7]
        num_inner_index = results[8]
        best_class = results[1]
        center_node = results[9]
        farthest_node = results[10]
        # print(f"total_mean_loss : {mean_MAE}")
        # print(f"best_class : {results[1]}")
        # print(f"max_index : {max_index}")
        # print(f"real_max_index : {torch.max(results[3].int())}")
        # print(f"num_inside : {results[5]}")
        # print(f"num_outside : {results[6]}")


    return encode_result



def save_dense_fast(index):

    index = index.to('cpu')

    if int(torch.max(index.int())) == 0:
        uint_i = 1
    else:
        uint_i = int(torch.max(index.int())).bit_length()

    t1s = time.perf_counter()
    save_results, padding_bits = convert_to_uint8_tensor_optimized(index, uint_i)
    t1e = time.perf_counter()
    t1 = t1e - t1s

    t2s = time.perf_counter()
    index_decode = restore_from_uint8_tensor(save_results, uint_i, padding_bits)
    t2e = time.perf_counter()
    t2 = t2e - t2s

    if not torch.equal(index_decode.int().to('cpu'), index.int().to('cpu')):
        raise NotImplementedError("Something wrong in save_dense function.")


    return save_results, torch.tensor(uint_i).to(dtype=torch.uint8), torch.tensor(padding_bits).to(dtype=torch.uint8)



def convert_to_uint8_tensor_optimized(arr, uint_i):

    arr = torch.as_tensor(arr, dtype=torch.int64)


    shifts = torch.arange(uint_i-1, -1, -1, device=arr.device)


    bits = (arr.unsqueeze(-1) >> shifts) & 1


    flattened = bits.flatten().to(torch.uint8)
    total_bits = flattened.numel()
    pad = (8 - (total_bits % 8)) % 8
    if pad != 0:
        padded = torch.cat([torch.zeros(pad, dtype=torch.uint8, device=arr.device), flattened])
    else:
        padded = flattened


    padded = padded.view(-1, 8)  # 将填充后的位数组重塑为 [n, 8]
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=arr.device)
    uint8_arr = (padded * weights).sum(dim=1).to(torch.uint8)

    return uint8_arr, pad




def restore_from_uint8_tensor(uint8_arr, bits_per_int, pad):

    bits = torch.bitwise_right_shift(uint8_arr.unsqueeze(-1), torch.arange(7, -1, -1, device=uint8_arr.device)) & 1
    bits = bits.flatten()


    if pad > 0:
        bits = bits[pad:]


    bits = bits.view(-1, bits_per_int)


    shifts = torch.arange(bits_per_int - 1, -1, -1, device=bits.device)
    restored_arr = (bits << shifts).sum(dim=1)

    return restored_arr





def get_Try_torch(tan_alpha, step_size, U, node_ld, rect_l):  # tan_alpha : 方向向量(非单位向量) ； step_size : 步长 ， n ：生成nodes的数量
    Allval = torch.arange(U).to(rect_l.device)
    Allval = Allval.reshape((-1,1))

    Cn = torch.tensor([rect_l/tan_alpha, rect_l]).to(rect_l.device)
    Cn = Cn/torch.linalg.norm(Cn)
    Cn = Cn.reshape((1,2))*step_size

    Try_array = Allval @ Cn
    Try_array = Try_array % rect_l
    Try_array = Try_array + node_ld

    x = Try_array[:, 0]
    y = Try_array[:, 1]


    return Try_array



def batch_compression_1d_torch(inputx_rect_inner, tree_rect_inner, Ba): ## reshape to 2d, and then reshape back to 1d


    # distance = torch.cdist(inputx_rect_inner, tree_rect_inner)
    # output_idx_node = torch.argmin(distance, dim=1).reshape(-1, 1)

    l = 500000
    output_idx_node = torch.tensor([]).to(inputx_rect_inner.device)
    while inputx_rect_inner.shape[0] != 0:
        distance_l = torch.cdist(inputx_rect_inner[:l,], tree_rect_inner)
        inputx_rect_inner = inputx_rect_inner[l:]
        output_idx_node_l = torch.argmin(distance_l, dim=1).reshape(-1, 1)
        output_idx_node = torch.cat((output_idx_node, output_idx_node_l), dim=0)

    cha = 0
    sizere = inputx_rect_inner.shape

    return output_idx_node, cha, sizere



def decompression_1d_torch(output_idx, Try):

    outputx = decompression_torch(output_idx, diction=Try)

    outputx = outputx.flatten()
    return outputx



def decompression_torch(output_idx, diction):

    outputx = diction[output_idx.int()]

    return outputx



def decompress_by_KDTree_torch(Try_rect_inner, tree_rect_inner, B):
    output_idx_node, cha_inner, size_tar_inner = batch_compression_1d_torch(B, tree_rect_inner, 128)

    # param_inner : 还原后的node
    B_star = decompression_1d_torch(output_idx_node, Try_rect_inner)
    index = output_idx_node

    return B_star, index



def test_ClassLoss_torch(Try_rect_inner, tree_rect_inner, inputx_rect_outer, rect_l1, each_dis, center_node, farthest_node, num_class, inner_MAE_loss):
    """
    测试将outer nodes分为 num_class个类别后的loss

    Returns:
        tensor_loss : MAE loss of back
        start_class : 从第几份开始分配index, 0,1,2, ...

    """
    farthest_dis = torch.linalg.norm(torch.abs(farthest_node - center_node))
    start_class = num_class
    dis_list = torch.tensor([(i+1) * each_dis.item() + (rect_l1.item()/2) for i in range(num_class)], dtype=torch.float32).to(farthest_dis.device)
    dis_list[-1] = farthest_dis
    factor_list = (rect_l1/2)/dis_list



    ##################################################
    ###############  Method 2 ########################
    center_node_tile = torch.tile(center_node, (inputx_rect_outer.shape[0], 1))
    dis_tile = torch.linalg.norm(inputx_rect_outer - center_node_tile, axis=1).reshape(-1,1)
    compare_dis_bool = dis_tile <= torch.tile(dis_list, (dis_tile.shape[0],1))
    node_class_1 = torch.argmax(compare_dis_bool.int(), axis=1) + 1
    node_factor_1 = factor_list[node_class_1 - 1].reshape(-1,1)
    OC_1 = inputx_rect_outer - torch.tile(center_node, (inputx_rect_outer.shape[0],1))
    OB_1 = OC_1 * node_factor_1
    B_1 = OB_1 + torch.tile(center_node, (inputx_rect_outer.shape[0],1))
    B_star_1, index_1 = decompress_by_KDTree_torch(Try_rect_inner, tree_rect_inner, B_1)
    B_star_1 = B_star_1.reshape(-1,2)
    OB_star_1 = B_star_1 - torch.tile(center_node, (inputx_rect_outer.shape[0],1))
    OC_star_1 = OB_star_1 / node_factor_1
    C_star_1 = OC_star_1 + torch.tile(center_node, (inputx_rect_outer.shape[0],1))

    tensor_loss_1 = torch.cat((inner_MAE_loss, torch.abs(C_star_1.flatten() - inputx_rect_outer.flatten())), dim=0)
    MAE_tensor_loss_1 = torch.mean(tensor_loss_1)
    ###############  Method 2 ########################
    ##################################################

    return MAE_tensor_loss_1, start_class





def find_inner_outer_torch(inputx, rect_l1):  # 以 inputx的质心为中心，rect_l1为inner正方形边长
    K = 2
    cha = len(inputx)
    newch = torch.ceil(torch.tensor([cha / K])).to(torch.int)

    pad_size = newch * K - cha
    # if pad_size.item() > 0:
    #     # inputx = np.pad(inputx, (0, pad_size), constant_values=np.mean(inputx))
    #     inputx = F.pad(inputx, (0, pad_size.item()), values=torch.mean(inputx))

    nodes = inputx.reshape(-1,2)
    x_node = nodes[:,0]
    y_node = nodes[:,1]
    x_center = torch.mean(x_node)
    y_center = torch.mean(y_node)
    center_node = torch.tensor([x_center, y_center]).to(rect_l1.device) # center

    # 定义矩形的左下角坐标 (x, y)，以及矩形的宽度和高度
    x_ld, y_ld = center_node[0] - rect_l1 / 2, center_node[1] - rect_l1 / 2
    x_lu, y_lu = center_node[0] - rect_l1 / 2, center_node[1] + rect_l1 / 2
    x_ru, y_ru = center_node[0] + rect_l1 / 2, center_node[1] + rect_l1 / 2
    x_rd, y_rd = center_node[0] + rect_l1 / 2, center_node[1] - rect_l1 / 2


    dis_matrix = torch.linalg.norm(torch.abs(nodes - center_node), axis=1)

    farthest_dis_matrix = torch.max(dis_matrix)


    farthest_node_matrix = nodes[torch.where(dis_matrix == farthest_dis_matrix)]


    check_inner = (nodes[:,0] >= x_lu) & (nodes[:,0] <= x_ru) & (nodes[:,1] >= y_ld) & (nodes[:,1] <= y_lu)
    inner_nodes_index = torch.where(check_inner == True)
    outer_nodes_index = torch.where(check_inner == False)

    inputx_rec_inner_matrix = nodes[inner_nodes_index]
    inputx_rec_outer_matrix = nodes[outer_nodes_index]


    return nodes, inputx_rec_inner_matrix, inputx_rec_outer_matrix, center_node, farthest_dis_matrix, farthest_node_matrix, pad_size




def compress_decom_v3(inputx, tensor_name, rect_l1, num_inner_list, class_max, loss_max,
                      loss_hope, device):  ## sparse matrix #most easy form to deal with all shapes, only need to deal with 1d
    """
    inputx : ori_param of i-layer
    rect_l1 : inner side
    num_inner_list : 可选的num_inner的list
    """
    inputx = inputx.to(device)
    rect_l1 = rect_l1.to(device)
    num_inner_list = num_inner_list.to(device)
    class_max = class_max.to(device)
    loss_max = loss_max.to(device)
    loss_hope = loss_hope.to(device)

    origin_inputx = inputx

    plt.clf()
    ori_shape = inputx.shape



    for num_inner in num_inner_list:

        d_inner = rect_l1 / (torch.ceil(torch.sqrt(num_inner)))
        l1 = torch.sqrt(rect_l1 ** 2 + d_inner ** 2)
        tan_alpha = torch.ceil(torch.sqrt(num_inner))
        sin_alpha = rect_l1 / l1
        step_size_inner = (rect_l1 / sin_alpha) / torch.ceil(torch.sqrt(num_inner))


        padding_nodes, inputx_rect_inner, inputx_rect_outer, center_node, farthest_dis, farthest_node, pad_size = find_inner_outer_torch(
            inputx, rect_l1)


        x_ld_inner, y_ld_inner = center_node[0] - rect_l1 / 2, center_node[1] - rect_l1 / 2
        x_lu_inner, y_lu_inner = center_node[0] - rect_l1 / 2, center_node[1] + rect_l1 / 2
        x_ru_inner, y_ru_inner = center_node[0] + rect_l1 / 2, center_node[1] + rect_l1 / 2
        x_rd_inner, y_rd_inner = center_node[0] + rect_l1 / 2, center_node[1] - rect_l1 / 2
        width_inner, height_inner = rect_l1, rect_l1

        ##################################
        #########  inner KDTree  #########
        Try_rect_inner = get_Try_torch(tan_alpha,
                                 step_size=step_size_inner,
                               U=num_inner,
                               node_ld=center_node - torch.tensor([rect_l1 / 2, rect_l1 / 2]).to(center_node.device),
                               rect_l=rect_l1).reshape([-1, 2])

        tree_rect_inner = Try_rect_inner
        #########  inner KDTree  #########
        ##################################

        ###########################################
        ############## inner MAE_loss #############
        inner_MAE_loss = torch.tensor([]).to(device)

        if inputx_rect_inner.numel() != 0:

            output_idx_node, cha_inner, size_tar_inner = batch_compression_1d_torch(inputx_rect_inner,
                                                                              tree_rect_inner, 128)


            param_inner_list = decompression_1d_torch(output_idx_node, Try_rect_inner)
            inner_MAE_loss = torch.abs(param_inner_list - inputx_rect_inner.flatten())
        ############## inner MAE_loss #############
        ###########################################

        ########################################
        ############## best_class  #############

        if inputx_rect_outer.numel() == 0:
            best_class = 0
            best_MAE_tensor_loss = torch.mean(inner_MAE_loss)

        else:

            best_MAE_tensor_loss = 100
            best_class = None

            for num_class in tqdm(range(1, int(class_max.item()) + 1)):
                each_dis = (farthest_dis - (rect_l1.item() / 2)) / torch.tensor(num_class).to(device)

                try:
                    MAE_tensor_loss, start_class = test_ClassLoss_torch(Try_rect_inner, tree_rect_inner, inputx_rect_outer,
                                                                  rect_l1, each_dis, center_node, farthest_node,
                                                                  num_class,
                                                                  inner_MAE_loss)

                    if MAE_tensor_loss <= best_MAE_tensor_loss:
                        best_MAE_tensor_loss = MAE_tensor_loss.item()
                        best_class = num_class

                    if MAE_tensor_loss <= loss_hope:
                        best_class = num_class
                        best_MAE_tensor_loss = MAE_tensor_loss.item()
                        break

                    if num_class == class_max:
                        each_dis = (farthest_dis - (rect_l1.item() / 2)) / torch.tensor(best_class).to(device)

                        MAE_tensor_loss, start_class = test_ClassLoss_torch(Try_rect_inner, tree_rect_inner,
                                                                      inputx_rect_outer, rect_l1, each_dis, center_node,
                                                                      farthest_node, best_class, inner_MAE_loss)
                        best_MAE_tensor_loss = MAE_tensor_loss.item()

                except:
                    print(f"{tensor_name} : class BUG!")

        ############## best_class  #############
        ########################################

        if best_MAE_tensor_loss > loss_max and torch.where(num_inner_list == num_inner.item())[0].item() != len(num_inner_list) - 1:
            continue
        else:
            break



    #############################################################################
    ########################### Encoding & Decoding ( Method 2 ) ################
    if best_class == 0:

        output_idx_node_1, cha_inner_1, size_tar_inner_1 = batch_compression_1d_torch(padding_nodes, tree_rect_inner, 128)
        new_param_1 = decompression_1d_torch(output_idx_node_1, Try_rect_inner)
        output_idx_node_1 = output_idx_node_1.flatten()

    else:

        check_inner_1 = ((padding_nodes[:, 0] >= x_lu_inner) & (padding_nodes[:, 0] <= x_ru_inner) &
                         (padding_nodes[:, 1] >= y_ld_inner) & (padding_nodes[:, 1] <= y_lu_inner))
        check_inner_1_index = torch.where(check_inner_1 == True)[0]

        center_node_tile = torch.tile(center_node, (padding_nodes.shape[0], 1))

        each_dis = (farthest_dis - (rect_l1.item() / 2)) / torch.tensor(best_class).to(device)

        dis_list = torch.linspace(0, best_class, best_class + 1) * each_dis.item() + (rect_l1.item() / 2)
        dis_list[-1] = farthest_dis
        dis_list = dis_list.to(device)

        factor_list = (rect_l1 / 2) / dis_list

        dis_tile = torch.linalg.norm(padding_nodes - center_node_tile, axis=1).reshape(-1, 1)
        compare_dis_bool = dis_tile <= torch.tile(dis_list, (dis_tile.shape[0], 1))
        check_class_1 = torch.argmax(compare_dis_bool.int(), axis=1)
        check_class_1[check_inner_1_index] = 0


        node_factor_1 = factor_list[check_class_1].reshape(-1, 1)
        OC_1 = padding_nodes - torch.tile(center_node, (padding_nodes.shape[0], 1))
        OB_1 = OC_1 * node_factor_1
        B_1 = OB_1 + torch.tile(center_node, (padding_nodes.shape[0], 1))
        # index_1 : inner中的index
        B_star_1, index_1 = decompress_by_KDTree_torch(Try_rect_inner, tree_rect_inner, B_1)
        B_star_1 = B_star_1.reshape(-1, 2)
        OB_star_1 = B_star_1 - torch.tile(center_node, (padding_nodes.shape[0], 1))
        OC_star_1 = OB_star_1 / node_factor_1
        C_star_1 = OC_star_1 + torch.tile(center_node, (padding_nodes.shape[0], 1))
        new_param_1 = C_star_1.flatten()
        output_idx_node_1 = index_1.flatten() + num_inner * check_class_1
        output_idx_node_1 = output_idx_node_1.flatten()

    ########################### Encoding & Decoding ( Method 2 ) ################
    #############################################################################

    new_param_1 = new_param_1.reshape(ori_shape)
    best_MAE_tensor_loss_list = torch.abs(new_param_1.flatten() - origin_inputx.flatten())

    max_value = output_idx_node_1.int().max().item()

    if max_value <= 255:
        dtype = torch.uint8
    elif max_value <= 65535:
        dtype = torch.uint16
    elif max_value <= 4294967295:
        dtype = torch.uint32
    else:
        dtype = torch.uint64

    # 将张量转换为目标 dtype
    output_idx_node_1 = output_idx_node_1.to(dtype)
    print(f"num_inner : {num_inner}")



    return (best_MAE_tensor_loss_list,
            best_class,
            new_param_1,
            output_idx_node_1,
            num_inner,
            inputx_rect_inner.shape[0],
            inputx_rect_outer.shape[0],
            pad_size,
            torch.where(num_inner_list == num_inner)[0].item(),
            center_node,
            farthest_node[0]
            )



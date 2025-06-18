import torch
import time
import os
import shutil
from .lib.Compress_Params_Standard_uint_i_ClassCenterHyper_Multi_VersionC_matrix_CP_best import *

def hf_encode(model, args):
    model_name = args['model_name']
    rect_l = args['rect_l']
    num_inner_list = args['num_inner_list']
    class_max = args['class_max']
    loss_max = args['loss_max']
    loss_hope = args['loss_hope']
    num_cores = args['num_cores']
    stop_threshold = args['stop_threshold']

    # encoding save path
    Save_Param_Path = f'./compressed_result/{model_name}_l_{str(rect_l)[0] + str(rect_l)[2:]}/'

    # # decoding files path
    # Decode_Param_Path = f'./compressed_result/{model_name}_l_{str(rect_l)[0] + str(rect_l)[2:]}'


    # 若存在，则删除重建
    if os.path.exists(Save_Param_Path):
        shutil.rmtree(Save_Param_Path)
        os.makedirs(Save_Param_Path, exist_ok=True)
    else:
        os.makedirs(Save_Param_Path, exist_ok=True)

    # 原参数保存地址
    Save_OriParam_Path = Save_Param_Path + 'Origin_Params.pth'
    # SpeedUp + FineTuning 后的params压缩结果的文件夹root地址
    Save_CompressedResult_RootPath = Save_Param_Path + 'Compressed_Dir/'
    # 压缩后再还原的params保存地址
    Save_BackParam_Path = Save_Param_Path + 'Back_Params.pth'
    Save_BackParam_Path_bin = Save_Param_Path + 'pytorch_model_back.bin'

    ## 保存 model的原参数 ##
    torch.save(model.state_dict(), Save_OriParam_Path)

    t1_start = time.perf_counter()
    ### model ：参数复原后的model ###
    size_result, model = compress_params(model, Save_CompressedResult_RootPath, rect_l, num_inner_list, class_max,
                                         loss_max, loss_hope, num_cores, stop_threshold)  # 返回 压缩后的文件大小
    t1_end = time.perf_counter()

    # 保存"Back_Params.pth"和"pytorch_model.bin"
    torch.save(model.state_dict(), Save_BackParam_Path)
    # torch.save(model.state_dict(), Save_BackParam_Path_bin)

    print(f"原参数大小为 {os.path.getsize(Save_OriParam_Path)}字节")
    print(f"压缩结果的大小为 {size_result}字节")
    print(f"压缩倍数{os.path.getsize(Save_OriParam_Path) / size_result}倍")

    print("Encoding Finished!!!!!!!")
    print(f"Compression Time : {t1_end - t1_start}秒 = {(t1_end - t1_start) / 60}分钟")


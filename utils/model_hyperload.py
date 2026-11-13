import gc
from utils.model_decode import decompress_params
import torch
from tqdm import tqdm
from torch import nn
import time

# ############################ Debug Setting ##################################
# import os
# from IPython.core.debugger import set_trace
#
# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported
#
# def check_tensors_gpu_ready(*tensors):
#     for t in tensors:
#         assert t.is_contiguous, "A tensor is not contiguous"
#         if not os.environ.get('TRITON_INTERPRET') == '1': assert t.is_cuda, "A tensor is not on cuda"
#
# def test_pid_conds(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
#     '''Test if condition on pids are fulfilled
#     E.g.:
#         '=0'  checks that pid_0 == 0
#         ',>1' checks that pid_1 > 1
#         '>1,=0' checks that pid_0 > 1 and pid_1 == 0
#     '''
#     pids = pid_0[0], pid_1[0], pid_2[0]
#     conds = conds.replace(' ','').split(',')
#     for i, (cond, pid) in enumerate(zip(conds, pids)):
#         if cond=='': continue
#         op, threshold = cond[0], int(cond[1:])
#         if op not in ['<','>','>=','<=','=', '!=']: raise ValueError(f"Rules may only use these ops: '<','>','>=','<=','=', '!='. Invalid rule: '{condition}'.")
#         op = '==' if op == '=' else op
#         if not eval(f'{pid} {op} {threshold}'): return False
#     return True
#
# assert test_pid_conds('')
# assert test_pid_conds('>0', [1], [1])
# assert not test_pid_conds('>0', [0], [1])
# assert test_pid_conds('=0,=1', [0], [1], [0])
#
# def breakpoint_if(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
#     '''Stop kernel, if any condition of pids is fulfilled'''
#     if test_pid_conds(conds, pid_0, pid_1, pid_2): set_trace()
#
# def print_if(txt, conds, pid_0=[0], pid_1=[0], pid_2=[0]):
#     '''Print txt, if any condition of pids is fulfilled'''
#     if test_pid_conds(conds, pid_0, pid_1, pid_2): print(txt)
#
# def cdiv(a,b): return (a + b - 1) // b
# assert cdiv(10,2)==5
# assert cdiv(10,3)==4
#
# ############################ Debug Setting ##################################

import triton
import triton.language as tl


def load_from_hyperencoded(model, hyper_dict, device):
    """
    model : the model of SAM
    hyper_dict : the decoded tensors dict by using 'hyper_op' decoded_mode
    """

    hyper_linear_module = HyperLinearV1
    # hyper_linear_module = nn.Linear(100, 100, bias=True)

    # Using HyperLinearV1() layers to replace nn.Linear()
    # Get all blocks # 获取所有 blocks
    blocks = []
    for name, module in model.named_children():
        blocks.append((name, module))

    # model = blocks; block = layers
    for i in tqdm(range(len(blocks))):
        name_block = blocks[i][0]  # 当前block的name
        block = blocks[i][1]  # 每个block

        for l_name, l in block.named_modules():
            if isinstance(l, nn.Linear):
                if l_name != '':
                    if hyper_dict[name_block + '.' + l_name + '.weight.load_type'] == 2:
                        hyper_linear = hyper_linear_module.from_linear(l, init_only=True)
                        # hyper_linear = hyper_linear_module
                        hyper_linear.to(next(block.parameters()).device)
                        replace_op_by_hyper(block, l_name, hyper_linear)
                else:
                    if hyper_dict[name_block + '.weight.load_type'] == 2:
                        hyper_linear = hyper_linear_module.from_linear(l, init_only=True)
                        hyper_linear.to(next(block.parameters()).device)
                        replace_op_by_hyper(block, l_name, hyper_linear)

        torch.cuda.empty_cache()
        gc.collect()

    for i in set(model.state_dict().keys()):
        if 'lm_head' in i:
            print(i)

    try:
        model.tie_weights()
    except:
        pass

    for name in set(hyper_dict.keys()):
        if '.load_type' in name:
            del hyper_dict[name]

    # Materialize meta parameters and HyperLinear buffers directly. The legacy
    # bundled Accelerate loader imports CuPy, which this inference path does not use.
    model.load_state_dict(hyper_dict, strict=True, assign=True)
    model.to(device)

    return model


def replace_op_by_hyper(block, name, hyper_linear):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = block
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], hyper_linear)

    else:
        setattr(block, name, hyper_linear)



class HyperLinearV1(nn.Module):
        def __init__(
                self, in_features, out_features, bias, dev, training=False
        ):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.training = training


            if self.in_features % 2 != 0:
                self.in_features += 1


            self.register_buffer(   # to define aux params
                "index_matrix",
                torch.zeros(
                    (self.out_features, int(self.in_features/2)),
                    dtype=torch.uint16,
                    device=dev,
                ),
            )

            self.register_buffer(  # to define aux params
                "each_dis",
                torch.zeros(
                    (1),
                    dtype=torch.float32,
                    device=dev,
                ),
            )

            self.register_buffer(  # to define aux params
                "U",
                torch.zeros(
                    (1),
                    dtype=torch.int64,
                    device=dev,
                ),
            )

            self.register_buffer(  # to define aux params
                "rect_l",
                torch.zeros(
                    (1),
                    dtype=torch.float32,
                    device=dev,
                ),
            )

            self.register_buffer(  # to define aux params
                "K",
                torch.zeros(
                    (1),
                    dtype=torch.uint8,
                    device=dev,
                ),
            )

            self.register_buffer(  # to define aux params
                "node_ld",
                torch.zeros(
                    (2),
                    dtype=torch.float32,
                    device=dev,
                ),
            )

            self.register_buffer(  # to define aux params
                "a_jump",
                torch.zeros(
                    (2),
                    dtype=torch.float32,
                    device=dev,
                ),
            )

            self.register_buffer(  # to define aux params
                "center",
                torch.zeros(
                    (2),
                    dtype=torch.float32,
                    device=dev,
                ),
            )


            if bias:
                self.register_buffer(
                    "bias",
                    torch.zeros(
                        (out_features),
                        dtype=torch.float32,  # origin : torch.float16
                        device=dev,
                    ),
                )
            else:
                self.bias = None

        @classmethod
        def from_linear(
                cls, linear, init_only=False
        ):
            hyper_linear = cls(  # input * param
                linear.in_features,  # param.shape[0]
                linear.out_features,  # param.shape[1]
                linear.bias is not None,
                linear.weight.device,
            )
            if init_only:  # just prepare for loading sd
                return hyper_linear


        def forward(self, x):
            out_shape = x.shape[:-1] + (self.out_features,)

            input_dtype = x.dtype
            dtype_output = x.dtype


            if self.training:
                out = HyperLinearMMFunction.apply(
                    self.index_matrix,
                    x,
                    dtype_output,
                    self.each_dis,
                    self.U,
                    self.rect_l,
                    self.K,
                    self.node_ld,
                    self.a_jump,
                    self.center,
                    self.bias,
                    self.out_features
                )
            else:
                with torch.no_grad():
                    ts = time.perf_counter()
                    out = HyperLinearMMFunction.apply(
                        self.index_matrix,
                        x,
                        dtype_output,
                        self.each_dis,
                        self.U,
                        self.rect_l,
                        self.K,
                        self.node_ld,
                        self.a_jump,
                        self.center,
                        self.bias,
                        self.out_features
                    )
                    te = time.perf_counter()
                    t = te - ts

            if input_dtype != torch.float16:
                out = out.to(dtype=input_dtype)

            return out.reshape(out_shape)

        def extra_repr(self) -> str:
            return (
                "in_features={}, out_features={}, bias={}".format(
                    self.in_features,
                    self.out_features,
                    self.bias is not None
                )
            )




from torch.autograd import Function

if torch.cuda.is_available():
    TRITON_AVAILABLE = True
else:
    TRITON_AVAILABLE = False


class HyperLinearMMFunction(Function):

    @staticmethod
    def forward(
            ctx,
            index_matrix,
            x,
            dtype_output,
            each_dis,
            U,
            rect_l,
            K,
            node_ld,
            a_jump,
            center,
            bias=None,
            out_features=0,
        ):

        out_shape = x.shape[:-1] + (out_features,)


        if TRITON_AVAILABLE:

            input_T = x.reshape(-1, x.shape[-1]).T

            out = triton_matmul_M3_10(
                index_matrix,
                input_T,  # input transpose
                dtype_output,
                each_dis.item(),
                float(U.item()),
                rect_l.item(),
                K.item(),
                node_ld,
                a_jump,
                center
            )


        else:
            raise NotImplementedError("Triton needed to be installed correctly, please check it.")

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)

        # always want 3D tensor if tensor is 2D
        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        return out







def triton_matmul_M3_10(
        index_matrix, # index_matrx
        input, # input
        dtype_output,
        each_dis,
        U,
        rect_l,
        K,
        node_ld,
        a_jump,
        center,
):

    M = index_matrix.shape[0]
    K_1 = index_matrix.shape[1]

    K_2 = input.shape[0]

    N = input.shape[1]
    result = torch.empty((M, N), device=index_matrix.device, dtype=dtype_output)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )


    assert node_ld.is_contiguous()
    assert a_jump.is_contiguous()
    assert center.is_contiguous()

    matmul_kernel_M3_10[grid](
        index_matrix,
        input,
        result,
        M,
        N,
        K_1,
        K_2,
        index_matrix.stride(0),
        index_matrix.stride(1),
        input.stride(0),
        input.stride(1),
        result.stride(0),
        result.stride(1),
        each_dis=each_dis,
        U=U,
        rect_l=rect_l,
        K=K,
        node_ld_ptr=node_ld,
        a_jump_ptr=a_jump,
        center_ptr=center,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=2,
    )

    return result.T



@triton.jit
def matmul_kernel_M3_10(
        a_ptr, b_ptr, c_ptr,
        M, N, K_1, K_2,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        each_dis: tl.constexpr, 
        U: tl.constexpr,
        rect_l: tl.constexpr,
        K: tl.constexpr,
        node_ld_ptr,
        a_jump_ptr,
        center_ptr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):

    rect_l_half = rect_l / 2.0
    each_dis_div_rect = each_dis / rect_l_half  
    U_INT: tl.constexpr = int(U)
    inv_rect_l: tl.constexpr = 1.0 / rect_l  


    node_ld_0 = tl.load(node_ld_ptr)
    node_ld_1 = tl.load(node_ld_ptr + 1)
    a_jump_0 = tl.load(a_jump_ptr)
    a_jump_1 = tl.load(a_jump_ptr + 1)
    center_0 = tl.load(center_ptr)
    center_1 = tl.load(center_ptr + 1)


    offset_0 = node_ld_0 - center_0
    offset_1 = node_ld_1 - center_1

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_am < M
    mask_n = offs_bn < N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    offs_k_b0 = offs_k * 2
    offs_k_b1 = offs_k * 2 + 1
    b_ptrs_0 = b_ptr + (offs_k_b0[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b_ptrs_1 = b_ptr + (offs_k_b1[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if K != 0: 
        for k_idx in range(0, tl.cdiv(K_1, BLOCK_SIZE_K)):
            mask_k = offs_k < (K_1 - k_idx * BLOCK_SIZE_K)
            mask_a = mask_m[:, None] & mask_k[None, :]
            mask_b = mask_k[:, None] & mask_n[None, :]


            b_0_f16 = tl.load(b_ptrs_0, mask=mask_b, other=0.0).to(tl.float16)
            b_1_f16 = tl.load(b_ptrs_1, mask=mask_b, other=0.0).to(tl.float16)
            

            theta_int = tl.load(a_ptrs, mask=mask_a, other=0).to(tl.int32)
            

            index_class = (theta_int // U_INT).to(tl.float32)
            theta_f32 = (theta_int % U_INT).to(tl.float32)

            X_0 = a_jump_0 * theta_f32
            X_1 = a_jump_1 * theta_f32
            mod_0_f32 = X_0 - tl.floor(X_0 * inv_rect_l) * rect_l 
            mod_1_f32 = X_1 - tl.floor(X_1 * inv_rect_l) * rect_l
            
            inv_node_factor = tl.fma(index_class, each_dis_div_rect, 1.0)
            C_0 = tl.fma((mod_0_f32 + offset_0), inv_node_factor, center_0)
            C_1 = tl.fma((mod_1_f32 + offset_1), inv_node_factor, center_1)
            
            # 【优化】C 降级为 FP16 并丢入点积
            accumulator += tl.dot(C_0.to(tl.float16), b_0_f16)
            accumulator += tl.dot(C_1.to(tl.float16), b_1_f16)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs_0 += BLOCK_SIZE_K * 2 * stride_bk
            b_ptrs_1 += BLOCK_SIZE_K * 2 * stride_bk

    else: 
        for k_idx in range(0, tl.cdiv(K_1, BLOCK_SIZE_K)):
            mask_k = offs_k < (K_1 - k_idx * BLOCK_SIZE_K)
            mask_a = mask_m[:, None] & mask_k[None, :]
            mask_b = mask_k[:, None] & mask_n[None, :]

            b_0_f16 = tl.load(b_ptrs_0, mask=mask_b, other=0.0).to(tl.float16)
            b_1_f16 = tl.load(b_ptrs_1, mask=mask_b, other=0.0).to(tl.float16)
            theta_int = tl.load(a_ptrs, mask=mask_a, other=0).to(tl.int32)
            
            theta_f32 = theta_int.to(tl.float32)

            X_0 = a_jump_0 * theta_f32
            X_1 = a_jump_1 * theta_f32
            
            mod_0_f32 = X_0 - tl.floor(X_0 * inv_rect_l) * rect_l
            mod_1_f32 = X_1 - tl.floor(X_1 * inv_rect_l) * rect_l
            
            B_0 = mod_0_f32 + node_ld_0
            B_1 = mod_1_f32 + node_ld_1
            
            accumulator += tl.dot(B_0.to(tl.float16), b_0_f16)
            accumulator += tl.dot(B_1.to(tl.float16), b_1_f16)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs_0 += BLOCK_SIZE_K * 2 * stride_bk
            b_ptrs_1 += BLOCK_SIZE_K * 2 * stride_bk

    c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

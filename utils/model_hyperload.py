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

    from sam_family.accelerate.big_modeling import load_checkpoint_and_dispatch_hyperload
    # loads the weights into modules
    load_checkpoint_and_dispatch_hyperload(
        model,
        checkpoint=hyper_dict,
        device_map={ "": device}, # balanced | auto | sequential
        max_memory=None,
        offload_folder=None,
        dtype=torch.float32,
    )

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

            # # Due to Matrix Transpose
            # M = self.out_features
            # N = self.in_features

            if self.in_features % 2 != 0:
                self.in_features += 1


            # quick sanity check (make sure aligment)
            # assert self.in_features % self.group_size == 0
            # assert out_features % (32 // self.w_bit) == 0

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
                    dtype=torch.int64, # 此为真实U值而非列表索引
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


            # if input_dtype != torch.float16:
            #     x = x.half()

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
        # The forward pass can use ctx.
        # ctx.save_for_backward(x, qweight, qzeros, scales, bias)
        # ctx.out_features = out_features

        out_shape = x.shape[:-1] + (out_features,)
        # x = x.to(torch.float16)


        if TRITON_AVAILABLE:
            # FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024

            # if FP16_MATMUL_HEURISTIC_CONDITION:
            #     out = awq_dequantize_triton(qweight, scales, qzeros)
            #     out = torch.matmul(x, out)
            # else:

            input_T = x.reshape(-1, x.shape[-1]).T.contiguous()

            out = triton_matmul_M3_2(
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


import triton
import triton.language as tl


def triton_matmul_M3_2(
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
        block_size_m: int = 64,
        block_size_n: int = 32,
        block_size_k: int = 32,
        group_size_m: int = 8
        # block_size_m: int = 32,
        # block_size_n: int = 32,
        # block_size_k: int = 16,
        # group_size_m: int = 8
):

    assert index_matrix.is_contiguous(), "矩阵A必须是连续的"
    assert input.is_contiguous(), "矩阵B必须是连续的"

    M = index_matrix.shape[0]
    K_1 = index_matrix.shape[1]

    K_2 = input.shape[0]

    N = input.shape[1]
    result = torch.empty((M, N), device=index_matrix.device, dtype=dtype_output)

    # 1D启动内核，每个块获得自己的程序。
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    # dis_list = torch.arange(0, K + 1) * each_dis + rect_l / 2
    # node_factor = ((rect_l / 2) / dis_list).to(index_matrix.device)

    assert node_ld.is_contiguous()
    assert a_jump.is_contiguous()
    assert center.is_contiguous()

    matmul_kernel_M3_2[grid](
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
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        num_stages=3,
        num_warps=4
    )

    return result.T  # 输出的是转置




@triton.jit
def matmul_kernel_M3_2(
        a_ptr, # a
        b_ptr, # b
        c_ptr, # c
        M, # M
        N, # N
        K_1, # K_1
        K_2, # K_2
        stride_am, # a.stride(0)
        stride_ak, # a.stride(1)
        stride_bk, # b.stride(0)
        stride_bn, # b.stride(1)
        stride_cm, # c.stride(0)
        stride_cn, # c.stride(1)
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
    """用于计算矩阵乘法C = A x B的内核。
    A的形状为(M, K)，B的形状为(K, N)且C的形状为(M, N)
    """
    # 映射程序id `pid`到它应该计算的C块。
    # 这是通过分组排序完成的，以促进L2数据重用。
    # 详见上方`L2缓存优化`部分。

    node_ld = tl.load(node_ld_ptr + tl.arange(0, 2))
    a_jump = tl.load(a_jump_ptr + tl.arange(0, 2))
    # a_jumps = tl.full((BLOCK_SIZE_M*BLOCK_SIZE_K, 1), 0, dtype=tl.float32) + a_jump
    center = tl.load(center_ptr + tl.arange(0, 2))

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 为A和B的第一个块创建指针。
    # 我们将在K方向移动时推进这个指针并累加
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    # offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_K)
    # offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    # offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    offs_k_1 = tl.arange(0, BLOCK_SIZE_K) # 单个AK块在a的K维度上的偏移值
    offs_k_2 = tl.arange(0, BLOCK_SIZE_K*2) # 单个BK块在b的K维度上的偏移值

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_1[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_2[:, None] * stride_bk + offs_bn[None, :] * stride_bn)


    # BLOCK_SIZE_a_index = 1
    # l1 = tl.full((4,1),0, dtype=tl.float32) + center
    # l2 = tl.reshape(tl.arange(1, 5), (4,1))
    # niubi = l1*l2
    # 迭代计算C矩阵的一个块。
    #####################################################################
    if K != 0 : # 若该index matrix已被分类
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K_1, BLOCK_SIZE_K)):
            b = tl.load(b_ptrs, mask=offs_k_2[:, None] < K_2 - k * BLOCK_SIZE_K * 2, other=0.0)
            theta = tl.cast(tl.load(a_ptrs, mask=offs_k_1[None, :] < K_1 - k * BLOCK_SIZE_K, other=0.0), dtype=tl.float32)
            theta = tl.reshape(theta,(BLOCK_SIZE_M*BLOCK_SIZE_K,1)) # flatten a to one column
            index_class = tl.floor(theta / U)
            dis_list = rect_l/2 + index_class * each_dis
            node_factor = (rect_l/2) / dis_list
            # theta_ = tl.where(index_class != 0, theta%(index_class*U), theta)
            theta_ = theta % U
            # B_star = (a_jump * theta_) % rect_l + node_ld
            # OB_star = B_star - center
            # OC_star = OB_star / node_factor
            # C_star = OC_star + center
            C_star = tl.where(node_factor != 1, ((a_jump * theta_) % rect_l + node_ld - center)/node_factor + center, (a_jump * theta_) %
                                                                                                                        rect_l + node_ld)
            a_decode = tl.reshape(C_star, (BLOCK_SIZE_M, BLOCK_SIZE_K*2))
            accumulator += tl.dot(a_decode, b)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * (stride_bk*2)



    else: # 若该index matrix 没有被分类
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K_1, BLOCK_SIZE_K)):
            theta = tl.cast(tl.load(a_ptrs, mask=offs_k_1[None, :] < K_1 - k * BLOCK_SIZE_K, other=0.0),
                            dtype=tl.float32)
            b = tl.load(b_ptrs, mask=offs_k_2[:, None] < K_2 - k * BLOCK_SIZE_K * 2, other=0.0)
            theta = tl.reshape(theta, (BLOCK_SIZE_M * BLOCK_SIZE_K, 1))  # flatten a to one column
            B_star = (a_jump * theta) % rect_l + node_ld
            a_decode = tl.reshape(B_star, (BLOCK_SIZE_M, BLOCK_SIZE_K * 2))
            accumulator += tl.dot(a_decode, b)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * (stride_bk * 2)

    c = accumulator.to(tl.float32)


    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)



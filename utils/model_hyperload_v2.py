"""Inference-only HyperLinear with a small pair codebook and fused CUDA GEMM.

The uint16 compressed matrices are retained. Only the possible decoded pairs
(not the dense weight matrices) are cached as FP16, matching v1's dot inputs.
"""

import torch
import triton
import triton.language as tl

from .model_hyperload import HyperLinearV1, load_from_hyperencoded


@triton.jit
def _build_codebook(OUT, DIST, U_PTR, RECT, CLASSES, NODE, JUMP, CENTER, SIZE,
                    BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    u = tl.load(U_PTR).to(tl.int32)
    rect = tl.load(RECT).to(tl.float32)
    k = tl.load(CLASSES).to(tl.int32)
    theta = (i % u).to(tl.float32)
    for j in tl.static_range(2):
        node = tl.load(NODE + j)
        jump = tl.load(JUMP + j)
        center = tl.load(CENTER + j)
        x = jump * theta
        mod = x - tl.floor(x * (1.0 / rect)) * rect
        if k != 0:
            scale = tl.fma((i // u).to(tl.float32), tl.load(DIST) / (rect / 2.0), 1.0)
            value = tl.fma(mod + (node - center), scale, center)
        else:
            value = mod + node
        tl.store(OUT + 2 * i + j, value, i < SIZE)


@triton.jit
def _lut_matmul(X, INDEX, LUT, BIAS, Y, M: tl.constexpr, N: tl.constexpr,
                K: tl.constexpr, SX0: tl.constexpr, SX1: tl.constexpr,
                HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr):
    # X [tokens, features], INDEX [outputs, features/2], Y [tokens, outputs].
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    pairs = tl.arange(0, BK // 2)
    components = tl.arange(0, 2)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + rk
        x = tl.load(X + rm[:, None] * SX0 + k[None, :] * SX1,
                    (rm[:, None] < M) & (k[None, :] < K), 0).to(tl.float16)
        kp = start * (BK // 2) + pairs
        idx = tl.load(INDEX + rn[:, None] * (K // 2) + kp[None, :],
                      (rn[:, None] < N) & (kp[None, :] < K // 2), 0).to(tl.int32)
        w = tl.load(LUT + idx[:, :, None] * 2 + components[None, None, :])
        w = tl.reshape(w, (BN, BK))
        acc = tl.dot(x, tl.trans(w), acc)
    # Match v1: cast GEMM output first, then add the original bias.
    value = acc.to(Y.dtype.element_ty).to(tl.float32)
    if HAS_BIAS:
        value += tl.load(BIAS + rn, rn < N, 0)[None, :]
    tl.store(Y + rm[:, None] * N + rn[None, :], value,
             (rm[:, None] < M) & (rn[None, :] < N))


class HyperLinearV1_v2(HyperLinearV1):
    """Drop-in inference replacement; compressed state_dict stays v1 compatible.

    Call prepare() after loading/changing buffers. No .item(), CPU/GPU scalar
    synchronization, autograd Function, or separate bias kernel in forward().
    """

    @torch.no_grad()
    def prepare(self):
        u, k = int(self.U.item()), int(self.K.item())
        if not self.index_matrix.is_cuda:
            raise ValueError('HyperLinearV1_v2 requires CUDA tensors.')
        if u <= 0 or k < 0:
            raise ValueError('Invalid HyperLinear codebook parameters.')
        lut = torch.empty((u * (k + 1), 2), device=self.index_matrix.device,
                          dtype=torch.float16)
        _build_codebook[(triton.cdiv(lut.shape[0], 256),)](
            lut, self.each_dis, self.U, self.rect_l, self.K, self.node_ld,
            self.a_jump, self.center, lut.shape[0], BLOCK=256)
        self.register_buffer('_codebook', lut, persistent=False)
        return self

    @classmethod
    def from_v1(cls, source):
        result = cls(source.in_features, source.out_features, source.bias is not None,
                     'meta')
        result.load_state_dict(source.state_dict(), strict=True, assign=True)
        return result.eval().prepare()

    @torch.no_grad()
    def forward(self, x):
        if not hasattr(self, '_codebook'):
            raise RuntimeError('Call prepare() after loading v2 weights.')
        if x.shape[-1] != self.in_features:
            if x.shape[-1] + 1 != self.in_features:
                raise ValueError('Input feature count does not match HyperLinear.')
            x = torch.nn.functional.pad(x, (0, 1))
        flat = x.reshape(-1, self.in_features)
        out = torch.empty((flat.shape[0], self.out_features), device=x.device, dtype=x.dtype)
        bm = 16 if flat.shape[0] < 32 else 64
        bn, bk = 64, 64
        _lut_matmul[(triton.cdiv(flat.shape[0], bm), triton.cdiv(self.out_features, bn))](
            flat, self.index_matrix, self._codebook, self.bias, out,
            flat.shape[0], self.out_features, self.in_features,
            flat.stride(0), flat.stride(1), self.bias is not None,
            BM=bm, BN=bn, BK=bk, num_warps=4, num_stages=2)
        return out.reshape(*x.shape[:-1], self.out_features)


def replace_hyperlinear_v2(model):
    """Convert an already-loaded v1 model without duplicating its weights."""
    count = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if type(child) is HyperLinearV1:
                setattr(parent, name, HyperLinearV1_v2.from_v1(child))
                count += 1
    return count


def load_from_hyperencoded_v2(model, hyper_dict, device):
    model = load_from_hyperencoded(model, hyper_dict, device)
    replace_hyperlinear_v2(model)
    return model

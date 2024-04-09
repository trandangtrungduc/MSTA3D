from typing import  List, Optional, Tuple
import math
import warnings

import torch
import torch.nn as nn

Tensor = torch.Tensor

def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    return torch._C._nn.linear(input, weight, bias)

def _get_softmax_dim(name: str, ndim: int, stacklevel: int) -> int:
    warnings.warn(
        "Implicit dimension choice for {} has been deprecated. "
        "Change the call to include dim=X as an argument.".format(name),
        stacklevel=stacklevel,
    )
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret

def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret

def _in_projection_packed(
    q: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    E = q.size(-1)
    return linear(q, w, b).chunk(3, dim=-1)

def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    talking_head: nn.Linear,
) -> Tuple[Tensor, Tensor]:
    B, Nt, E = v.shape
    q = q / math.sqrt(E)
    attn = torch.einsum("bhnd,bhdk->bhnk", q, k.transpose(-2, -1))
    attn = talking_head(attn.permute(0, 2, 3, 1))
    attn = softmax(attn, dim=2)
    attn = attn.permute(0, 3, 1, 2)
    attn = attn.contiguous().view(B, Nt, Nt)
    output = torch.einsum("bnk,bkd -> bnd", attn, v)
    return output, attn

def multi_head_attention_forward(
    query: Tensor,
    num_heads: int,
    talking_head: nn.Linear,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.shape
    head_dim = embed_dim // num_heads
    q, k, v = _in_projection_packed(query, in_proj_weight, in_proj_bias)

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

    q = q.view(bsz, num_heads, tgt_len, head_dim)
    k = k.view(bsz, num_heads, tgt_len, head_dim)

    attn_output, attn_output_weights = _scaled_dot_product_attention(
        q, k, v, talking_head)
    attn_output = attn_output.transpose(
        0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    attn_output_weights = attn_output_weights.view(
            bsz, num_heads, tgt_len, tgt_len)
    return attn_output, attn_output_weights.sum(dim=1) / num_heads

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads,bias=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.talking_head = nn.Linear(self.num_heads, self.num_heads)
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj_weight = nn.parameter.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_bias = nn.parameter.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        query = query.transpose(1, 0)
        attn_output, attn_output_weights = multi_head_attention_forward(query, self.num_heads, self.talking_head, self.in_proj_weight, self.in_proj_bias, self.out_proj.weight, self.out_proj.bias)
        return attn_output.transpose(1, 0), attn_output_weights
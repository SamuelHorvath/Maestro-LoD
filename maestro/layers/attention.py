import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.functional import pad

from .linear import MaestroLinear


class MaestroMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, batch_first=False):
        super(MaestroMultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding size needs to be divisible by num_heads"

        self.kdim = kdim if kdim else embed_dim
        self.vdim = vdim if vdim else embed_dim

        self.batch_first = batch_first

        # Define linear layers for Q, K, V
        self.q_proj = MaestroLinear(embed_dim, embed_dim, bias=bias)
        self.k_proj = MaestroLinear(self.kdim, embed_dim, bias=bias)
        self.v_proj = MaestroLinear(self.vdim, embed_dim, bias=bias)

        # Optionally add a bias to the key and value
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
            xavier_normal_(self.bias_k)
            xavier_normal_(self.bias_v)

        self.out_proj = MaestroLinear(embed_dim, embed_dim, bias=bias)

        self.dropout_layer = nn.Dropout(dropout)

        self.add_zero_attn = add_zero_attn

    @staticmethod
    def _canonical_mask(mask, target_type):
        if mask is None:
            return None
        if not torch.is_floating_point(mask):
            mask = (
                torch.zeros_like(mask, dtype=target_type)
                .masked_fill_(mask, float("-inf"))
            )
        return mask

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, sampler=None):

        if sampler is None:
            def none_sampler():
                return None
            sampler = none_sampler

        if self.batch_first:
            query, key, value = query.transpose(-3, -2), \
                key.transpose(-3, -2), value.transpose(-3, -2)

        tgt_len, bsz, embed_dim = query.size()
        src_len = key.shape[0]

        if attn_mask is not None:
            assert attn_mask.shape == (tgt_len, src_len), \
                f"expecting attn_mask shape of {(tgt_len, src_len)}," \
                f"but got {attn_mask.shape}"

        key_padding_mask = self._canonical_mask(key_padding_mask, query.dtype)
        attn_mask = self._canonical_mask(attn_mask, query.dtype)

        # Linear transformations
        q = self.q_proj(query, p=sampler())
        k = self.k_proj(key, p=sampler())
        v = self.v_proj(value, p=sampler())

        # add bias along batch dimension (currently second)
        bias_k = self.bias_k
        bias_v = self.bias_v
        if bias_k is not None and bias_v is not None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None

        # Fold heads into the batch dimension
        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).transpose(
            0, 2)
        k = k.view(-1, bsz, self.num_heads, self.head_dim).transpose(
            0, 2)
        v = v.view(-1, bsz, self.num_heads, self.head_dim).transpose(
            0, 2)

        # add zero attention along batch dimension
        if self.add_zero_attn:
            zero_attn_shape = (self.num_heads, bsz, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(
                zero_attn_shape, dtype=k.dtype, device=k.device)], dim=2)
            v = torch.cat([v, torch.zeros(
                zero_attn_shape, dtype=v.dtype, device=v.device)], dim=2)
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(2)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, " \
                f"but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, self.num_heads, -1, -1)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask
                attn_mask = attn_mask.transpose(0, 1)

        # Scaled dot-product attention
        scaled_q = q / self.head_dim**0.5
        attn_output_weights = scaled_q @ k.transpose(-2, -1)

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout_layer(attn_output_weights)

        attn_output = attn_output_weights @ v

        # Convert attention output back to its original size
        attn_output = attn_output.transpose(0, 2).contiguous().view(
            tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output, p=sampler())

        if self.batch_first:
            attn_output = attn_output.transpose(-3, -2)

        if need_weights:
            return attn_output, attn_output_weights.mean(dim=0)
        else:
            return attn_output

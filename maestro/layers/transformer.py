from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm

from .attention import MaestroMultiheadAttention
from .linear import MaestroLinear
from ..samplers import BaseSampler

__all__ = ['MaestroTransformer', 'MaestroTransformerEncoder',
           'MaestroTransformerDecoder', 'MaestroTransformerEncoderLayer',
           'MaestroTransformerDecoderLayer']


class MaestroTransformer(torch.nn.Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, norm_first: bool = False) -> None:

        if custom_encoder is None:
            encoder_layer = MaestroTransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation, layer_norm_eps, batch_first, norm_first)
            encoder_norm = LayerNorm(
                d_model, eps=layer_norm_eps)
            custom_encoder = MaestroTransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is None:
            decoder_layer = MaestroTransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation, layer_norm_eps, batch_first, norm_first)
            decoder_norm = LayerNorm(
                d_model, eps=layer_norm_eps)
            custom_decoder = MaestroTransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm)

        super().__init__(
            d_model, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, dropout, activation, custom_encoder,
            custom_decoder, layer_norm_eps, batch_first, norm_first)

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                sampler: BaseSampler = None) -> Tensor:
        if sampler is None:
            def none_sampler():
                return None
            sampler = none_sampler

        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
            sampler=sampler)
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            sampler=sampler)
        return output


class MaestroTransformerEncoder(torch.nn.TransformerEncoder):

    def __init__(self, encoder_layer, num_layers, norm=None,
                 enable_nested_tensor=True, mask_check=True):
        super().__init__(encoder_layer, num_layers, norm,
                         enable_nested_tensor, mask_check)

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None,
            sampler: BaseSampler = None) -> Tensor:

        if sampler is None:
            def none_sampler():
                return None
            sampler = none_sampler

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src

        # Prevent type refinement
        make_causal = (is_causal is True)

        if is_causal is None:
            if mask is not None:
                sz = mask.size(0)
                causal_comparison = torch.triu(
                    torch.ones(sz, sz, device=mask.device) * float('-inf'),
                    diagonal=1).to(mask.dtype)

                if torch.equal(mask, causal_comparison):
                    make_causal = True

        is_causal = make_causal

        for mod in self.layers:
            output = mod(
                output, src_mask=mask, is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask,
                sampler=sampler)

        if self.norm is not None:
            output = self.norm(output)

        return output


class MaestroTransformerDecoder(torch.nn.TransformerDecoder):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                sampler: BaseSampler = None) -> Tensor:
        if sampler is None:
            def none_sampler():
                return None
            sampler = none_sampler

        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         sampler=sampler)

        if self.norm is not None:
            output = self.norm(output)

        return output


class MaestroTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 norm_first: bool = False,
                 device=None, dtype=None) -> None:

        super().__init__(
            d_model, nhead, dim_feedforward, dropout, activation,
            layer_norm_eps, batch_first, norm_first, device, dtype)

        self.self_attn = MaestroMultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = MaestroLinear(d_model, dim_feedforward)
        self.linear2 = MaestroLinear(dim_feedforward, d_model)

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            sampler: BaseSampler = None) -> Tensor:
        if sampler is None:
            def none_sampler():
                return None
            sampler = none_sampler

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask,
                src_key_padding_mask, is_causal=is_causal,
                sampler=sampler)
            x = x + self._ff_block(self.norm2(x),
                                   sampler=sampler)
        else:
            x = self.norm1(x + self._sa_block(
                x, src_mask,
                src_key_padding_mask, is_causal=is_causal,
                sampler=sampler))
            x = self.norm2(x + self._ff_block(
                x, sampler=sampler))

        return x

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor],
                  is_causal: bool = False,
                  sampler: BaseSampler = None) -> Tensor:

        if sampler is None:
            def none_sampler():
                return None
            sampler = none_sampler

        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal,
                           sampler=sampler)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor, sampler: BaseSampler = None) -> Tensor:
        if sampler is None:
            def none_sampler():
                return None
            sampler = none_sampler

        x = self.linear2(
            self.dropout(self.activation(
                self.linear1(x, p=sampler()))),
            p=sampler())
        return self.dropout2(x)


class MaestroTransformerDecoderLayer(torch.nn.TransformerDecoderLayer):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 norm_first: bool = False,
                 device=None, dtype=None) -> None:
        super().__init__(
            d_model, nhead, dim_feedforward, dropout, activation,
            layer_norm_eps, batch_first, norm_first, device, dtype
        )
        self.self_attn = MaestroMultiheadAttention(
            d_model, nhead, dropout=dropout,
            batch_first=batch_first)
        self.multihead_attn = MaestroMultiheadAttention(
            d_model, nhead, dropout=dropout,
            batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = MaestroLinear(d_model, dim_feedforward)
        self.linear2 = MaestroLinear(dim_feedforward, d_model)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        sampler: BaseSampler = None,
    ) -> Tensor:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        if sampler is None:
            def none_sampler():
                return None
            sampler = none_sampler

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal,
                sampler=sampler)
            x = x + self._mha_block(
                self.norm2(x), memory, memory_mask,
                memory_key_padding_mask, memory_is_causal,
                sampler=sampler)
            x = x + self._ff_block(self.norm3(x), sampler=sampler)
        else:
            x = self.norm1(x + self._sa_block(
                x, tgt_mask, tgt_key_padding_mask, tgt_is_causal,
                sampler=sampler))
            x = self.norm2(x + self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask,
                memory_is_causal, sampler=sampler))
            x = self.norm3(x + self._ff_block(x, sampler=sampler))

        return x

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor],
                  is_causal: bool = False,
                  sampler: BaseSampler = None) -> Tensor:

        if sampler is None:
            def none_sampler():
                return None
            sampler = none_sampler

        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False,
                           sampler=sampler)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor],
                   is_causal: bool = False,
                   sampler: BaseSampler = None) -> Tensor:
        if sampler is None:
            def none_sampler():
                return None
            sampler = none_sampler

        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False,
                                sampler=sampler)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor, sampler: BaseSampler = None) -> Tensor:
        if sampler is None:
            def none_sampler():
                return None
            sampler = none_sampler

        x = self.linear2(
            self.dropout(self.activation(
                self.linear1(x, p=sampler()))),
            p=sampler())
        return self.dropout3(x)

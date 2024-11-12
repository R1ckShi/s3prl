import logging
import time
import kaldiio, os
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Iterable, Optional
from .sensevoice_fsq_components import *
    
att_type_dict = {
    # "default": MultiHeadAttention,
    # "sdpa": MultiHeadAttentionSdpa,
    # "self_att": MultiHeadAttentionRoPE,
    # "self_att_sdpa": MultiHeadAttentionSdpaRoPE,
    # "self_att_fsmn": MultiHeadAttentionFSMNRoPE,
    "self_att_fsmn_sdpa": MultiHeadAttentionFSMNSdpaRoPE,
}

class EncoderLayerSANMLarge(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()

        att_type = kwargs.get("att_type", "self_att_fsmn_sdpa")
        self.attn = att_type_dict[att_type](linear_units, attention_heads, **kwargs)
        self.attn_ln = LayerNorm(linear_units)

        n_mlp = linear_units * 4
        self.mlp = nn.Sequential(
            Linear(linear_units, n_mlp), nn.GELU(), Linear(n_mlp, linear_units)
        )
        self.mlp_ln = LayerNorm(linear_units)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        x = x + self.attn(self.attn_ln(x), mask=mask, **kwargs)[0]

        x = x + self.mlp(self.mlp_ln(x))
        return x

class SenseVoiceQuantizedEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        linear_units: int,
        attention_heads: int,
        num_blocks: int,
        quantize_layer_idx: int,
        normalized_quant_input: bool,
        quantizer_config: dict,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = Conv1d(input_size, linear_units, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv1d(linear_units, linear_units, kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList(
            [
                EncoderLayerSANMLarge(linear_units, attention_heads, **kwargs)
                for _ in range(num_blocks)
            ]
        )
        self.ln_post = LayerNorm(linear_units)
        self.use_padmask = kwargs.get("use_padmask", True)
        self.downsample_rate = kwargs.get("downsample_rate", 4)

        self.linear_units = linear_units
        self.quantize_layer_idx = quantize_layer_idx
        self.normalized_quant_input = normalized_quant_input
        self.quantizer = self.build_quantizer(quantizer_config)

    def build_quantizer(self, vq_config):
        if vq_config is None:
            logging.error("None vq_config")
        from omegaconf import OmegaConf, DictConfig
        if isinstance(vq_config, (OmegaConf, DictConfig)):
            vq_config = OmegaConf.to_container(vq_config)
        name = vq_config.pop("name", "costume_quantizer")
        from .finite_scalar_quantizer import FSQ
        quantizer = FSQ(
            input_size=self.linear_units,
            **vq_config,
        )
        vq_config["name"] = "finite_scalar_quantizer"
        return quantizer
        # if name == "costume_quantizer":
        #     # from funasr.models.sense_voice.quantizer.costume_quantizer import CostumeQuantizer
        #     from .quantizer.costume_quantizer import CostumeQuantizer  # fake
        #     quantizer = CostumeQuantizer(
        #         input_size=self.linear_units,
        #         **vq_config,
        #     )
        #     vq_config["name"] = "costume_quantizer"
        #     return quantizer
        # elif name == "lookup_free_quantizer":
        #     # from funasr.models.sense_voice.quantizer.lookup_free_quantizer import LFQ
        #     from .quantizer.lookup_free_quantizer import LFQ  # fake
        #     quantizer = LFQ(
        #         input_size=self.linear_units,
        #         **vq_config,
        #     )
        #     vq_config["name"] = "lookup_free_quantizer"
        #     return quantizer
        # elif name == "finite_scalar_quantizer":
        #     from .finite_scalar_quantizer import FSQ

        #     quantizer = FSQ(
        #         input_size=self.linear_units,
        #         **vq_config,
        #     )
        #     vq_config["name"] = "finite_scalar_quantizer"
        #     return quantizer
        # else:
        #     raise NotImplemented("quantizer {} not implemented".format(name))

    def quantize_enc_outs(self, x):
        ret_dict = {}

        if self.normalized_quant_input:
            x = F.normalize(x, dim=-1)
        ret_dict["quant_in"] = x
        x, indices, commit_loss, sub_quants = self.quantizer(x)
        ret_dict["quant_out"] = x
        ret_dict["indices"] = indices
        ret_dict["quant_loss"] = commit_loss

        return x, ret_dict

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor = None,
        **kwargs,
    ):
        use_padmask = self.use_padmask
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        only_extract_tokens = kwargs.get("only_extract_tokens", False)

        n_frames = x.size(1)
        max_pos = n_frames

        if ilens is not None:
            if self.downsample_rate == 4:
                olens = (
                    1
                    + (ilens - self.conv1.kernel_size[0] + 2 * self.conv1.padding[0])
                    // self.conv1.stride[0]
                )
            else:
                olens = ilens
            olens = (
                1
                + (olens - self.conv2.kernel_size[0] + 2 * self.conv2.padding[0])
                // self.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        if use_padmask and olens is not None:
            padding_mask = (~make_pad_mask(olens)[:, None, :]).to(torch.bool).to(x.device)
        else:
            padding_mask = None

        device = x.device
        seq_length = x.shape[1]
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        for layer, block in enumerate(self.blocks):
            x = block(x, mask=padding_mask, position_ids=position_ids)
            # if self.quantize_layer_idx is not None and self.quantizer is not None:
            #     if layer == self.quantize_layer_idx:
            #         hint_once(
            #             f"Quantization at layer {layer} wit {self.quantizer}",
            #             "normalize_quant_enc_out",
            #             rank=0,
            #         )
            #         x, ret_dict = self.quantize_enc_outs(x)
            #         if only_extract_tokens:
            #             return (x, ret_dict), olens

        # remove ln_post
        # x = self.ln_post(x)

        if ilens is None:
            return x
        else:
            return x, olens
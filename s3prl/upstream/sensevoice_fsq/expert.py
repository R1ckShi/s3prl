from collections import OrderedDict
from typing import Dict, List, Union

import yaml
import torch
import logging
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .sensevoice_fsq_encoder import SenseVoiceQuantizedEncoder

HIDDEN_DIM = 8


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        self.name = "sensevoice_fsq"

        print(
            f"{self.name} - You can use model_config to construct your customized model: {model_config}"
        )
        print(f"{self.name} - You can use ckpt to load your pretrained weights: {ckpt}")
        print(
            f"{self.name} - If you store the pretrained weights and model config in a single file, "
            "you can just choose one argument (ckpt or model_config) to pass. It's up to you!"
        )

        # load model_config
        with open(model_config, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        encoder_conf = config.get('encoder_conf', {})
        logging.warning("encoder conf: {}".format(encoder_conf))
        self.encoder = SenseVoiceQuantizedEncoder(**encoder_conf)
        
        from .whisper_frontend import WhisperFrontend
        frontend_conf = config.get("frontend_conf", {})
        logging.warning("frontend conf: {}".format(frontend_conf))
        self.frontend = WhisperFrontend(**frontend_conf)

        # base model1
        # /nfs/shixian.shi/workspace/models/funasr_results/asr/whisper/5m-8gpu/l6_fulldata_exp5_t0/ds-model.pt.ep1.340000
        if ckpt is not None:
            loaded = torch.load(ckpt)['state_dict']
            logging.warning("Loaded model from {}.".format(ckpt))
            for k in list(loaded.keys()):
                if 'decoder' in k or (k.startswith("model.encoder.blocks.") and int(k.split('.')[3]) > 5):
                    del loaded[k]
            for k in list(loaded.keys()):
                if k[14:] not in list(self.encoder.state_dict().keys()):
                    logging.warning("Delete key in encoder: {}".format(k))
                    del loaded[k]
                else:
                    loaded[k[14:]] = loaded[k]
                    del loaded[k]
                    logging.warning("Keeping key in encoder: {}".format(k[14:]))
            # logging.warning("Ready to load: {}".format(loaded))
            # import pdb; pdb.set_trace()
            self.encoder.load_state_dict(loaded)
        
        # The model needs to be a nn.Module for finetuning, not required for representation extraction
        # self.model1 = nn.Linear(1, HIDDEN_DIM)
        # self.model2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 640

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        wavs = pad_sequence(wavs, batch_first=True).unsqueeze(-1)
        wavs = wavs.squeeze(-1)
        wavs_lens = torch.tensor([wavs.shape[-1]]*wavs.shape[0])
        features, feature_lens = self.frontend(wavs, wavs_lens)
         
        encoder_out, encoder_out_lens = self.encoder(features, feature_lens)
        quantize_out = self.encoder.quantize_enc_outs(encoder_out)
        # codes = self.encoder.quantizer.indices_to_codes(quantize_out[1]['indices'])
        # import pdb; pdb.set_trace()
        hidden = encoder_out
        # hidden: (batch_size, max_len, hidden_dim)
        feature = encoder_out
        # feature: (batch_size, max_len, hidden_dim)
        # import pdb; pdb.set_trace()
        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        return {
            "hidden_states": [hidden, feature],
            "PR": [hidden, feature],
            "ASR": [hidden, feature],
            "QbE": [hidden, feature],
            "SID": [hidden, feature],
            "ASV": [hidden, feature],
            "SD": [hidden, feature],
            "ER": [hidden, feature],
            "SF": [hidden, feature],
            "SE": [hidden, feature],
            "SS": [hidden, feature],
            "secret": [hidden, feature],
        }

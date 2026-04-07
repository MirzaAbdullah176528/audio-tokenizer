from typing import Dict, Optional, Tuple, Any
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from configs.model_config import MimiConfig
from mimi.encoder import MimiEncoder
from mimi.decoder import MimiDecoder
from mimi.quantizer import SplitRVQ


class Mimi(nn.Module):
    config: MimiConfig
    dtype: jnp.dtype = jnp.bfloat16
    frozen_encoder: bool = False
    frozen_decoder: bool = False

    def setup(self):
        self.encoder = MimiEncoder(config=self.config.encoder, dtype=self.dtype)
        self.decoder = MimiDecoder(config=self.config.decoder, dtype=self.dtype)
        self.quantizer = SplitRVQ(config=self.config.quantizer, dtype=self.dtype)

    def encode(
        self,
        waveform: jnp.ndarray,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if self.frozen_encoder:
            z = jax.lax.stop_gradient(self.encoder(waveform, train=False))
        else:
            z = self.encoder(waveform, train)
        reconstructed_latent, tokens, commitment_loss, distillation_features = (
            self.quantizer(z, train)
        )
        return reconstructed_latent, tokens, commitment_loss, distillation_features

    def decode(
        self,
        latent: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        if self.frozen_decoder:
            return jax.lax.stop_gradient(self.decoder(latent, train=False))
        return self.decoder(latent, train)

    def decode_tokens(self, tokens: jnp.ndarray) -> jnp.ndarray:
        latent = self.quantizer.decode_tokens(tokens)
        return self.decoder(latent, train=False)

    def __call__(
        self,
        waveform: jnp.ndarray,
        train: bool = False,
        wavlm_targets: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        reconstructed_latent, tokens, commitment_loss, distillation_features = (
            self.encode(waveform, train)
        )
        reconstructed_audio = self.decode(reconstructed_latent, train)

        result = {
            "reconstructed_audio": reconstructed_audio,
            "tokens": tokens,
            "commitment_loss": commitment_loss,
            "distillation_features": distillation_features,
        }

        if wavlm_targets is not None:
            distillation_loss = self.quantizer.compute_distillation_loss(
                distillation_features, wavlm_targets
            )
            result["distillation_loss"] = distillation_loss

        return result



def load_mimi_pretrained_weights(
    mimi_model: Mimi,
    init_params: Dict,
    pytorch_checkpoint_path: str,
    device: str = "cpu",
) -> Dict:
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for loading pretrained Mimi weights. pip install torch")

    print(f"Loading pretrained Mimi weights from {pytorch_checkpoint_path}")
    import glob, os
    if os.path.isdir(pytorch_checkpoint_path):
        candidates = glob.glob(os.path.join(pytorch_checkpoint_path, "*.pt")) + \
            glob.glob(os.path.join(pytorch_checkpoint_path, "*.bin")) + \
            glob.glob(os.path.join(pytorch_checkpoint_path, "*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No .pt or .bin file found in {pytorch_checkpoint_path}")
        pytorch_checkpoint_path = candidates[0]
    state_dict = torch.load(pytorch_checkpoint_path, map_location=device)

    if "model" in state_dict:
        state_dict = state_dict["model"]

    converted = _convert_pytorch_to_flax(state_dict)
    print(f"Converted {len(converted)} parameter tensors from PyTorch checkpoint")
    return converted


def _convert_pytorch_to_flax(pt_state_dict: Dict) -> Dict:
    flax_params = {}

    for pt_key, tensor in pt_state_dict.items():
        arr = np.array(tensor.float().numpy()).astype(np.float32)

        if tensor.ndim == 3 and pt_key.endswith(".weight") and "conv" in pt_key.lower():
            arr = arr.transpose(2, 0, 1)
        elif tensor.ndim == 2 and pt_key.endswith(".weight") and "linear" in pt_key.lower():
            arr = arr.T

        flax_key = _map_pytorch_key_to_flax(pt_key)
        if flax_key is not None:
            flax_params[flax_key] = jnp.array(arr, dtype=jnp.bfloat16)

    return flax_params


def _map_pytorch_key_to_flax(pt_key: str) -> Optional[str]:
    key = pt_key
    key = key.replace("encoder.", "encoder/")
    key = key.replace("decoder.", "decoder/")
    key = key.replace("quantizer.", "quantizer/")
    key = key.replace(".", "/")
    return key


def create_mimi_for_finetuning(
    config: MimiConfig,
    pretrained_checkpoint: Optional[str] = None,
    freeze_encoder: bool = False,
    freeze_decoder: bool = False,
) -> Tuple[Mimi, Optional[Dict]]:
    mimi = Mimi(
        config=config,
        frozen_encoder=freeze_encoder,
        frozen_decoder=freeze_decoder,
    )
    pretrained_params = None
    if pretrained_checkpoint is not None:
        dummy_input = jnp.zeros((1, 1, config.sample_rate))
        key = jax.random.PRNGKey(0)
        variables = mimi.init(key, dummy_input)
        pretrained_params = load_mimi_pretrained_weights(
            mimi, variables["params"], pretrained_checkpoint
        )
    return mimi, pretrained_params

from typing import List, Optional, Tuple, Dict
import jax
import jax.numpy as jnp
import numpy as np

from configs.model_config import InnerMonologueConfig, MultiStreamConfig


FRAME_RATE = 12.5
SECONDS_PER_FRAME = 1.0 / FRAME_RATE


def align_text_to_frames(
    word_token_lists: List[List[int]],
    word_start_times: List[float],
    total_frames: int,
    config: InnerMonologueConfig,
    text_delay_seconds: float = 0.0,
) -> np.ndarray:
    sequence = np.full(total_frames, config.pad_token_id, dtype=np.int32)

    for i, (tokens, start_time) in enumerate(zip(word_token_lists, word_start_times)):
        adjusted_start = start_time + text_delay_seconds
        start_frame = int(adjusted_start * FRAME_RATE)
        start_frame = max(0, min(start_frame, total_frames - 1))

        if start_frame > 0 and sequence[start_frame - 1] == config.pad_token_id:
            sequence[start_frame - 1] = config.epad_token_id

        for j, token_id in enumerate(tokens):
            pos = start_frame + j
            if pos < total_frames:
                sequence[pos] = token_id

    return sequence


def build_inner_monologue_sequence(
    whisper_words: List[str],
    whisper_start_times: List[float],
    tokenizer,
    total_frames: int,
    config: InnerMonologueConfig,
    text_delay_seconds: float = 0.0,
) -> np.ndarray:
    word_token_lists = tokenizer.tokenize_word_list(whisper_words)
    return align_text_to_frames(
        word_token_lists,
        whisper_start_times,
        total_frames,
        config,
        text_delay_seconds,
    )


def randomize_text_delay(
    text_tokens: np.ndarray,
    word_token_lists: List[List[int]],
    word_start_times: List[float],
    total_frames: int,
    config: InnerMonologueConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    delay = rng.uniform(
        config.text_delay_pretrain_min,
        config.text_delay_pretrain_max,
    )
    return build_inner_monologue_sequence(
        [""] * len(word_token_lists),
        word_start_times,
        None,
        total_frames,
        config,
        text_delay_seconds=delay,
    )


class MultiStreamProcessor:
    def __init__(
        self,
        multistream_config: MultiStreamConfig,
        inner_monologue_config: InnerMonologueConfig,
    ):
        self.ms_config = multistream_config
        self.im_config = inner_monologue_config

    def build_joint_tokens_single_stream(
        self,
        moshi_audio_tokens: np.ndarray,
        text_tokens: np.ndarray,
        acoustic_delay: int,
        pad_token: int = 0,
    ) -> np.ndarray:
        S = moshi_audio_tokens.shape[0]
        Q = self.ms_config.num_codebooks
        K_total = self.ms_config.total_depth_streams

        joint = np.zeros((S, K_total), dtype=np.int32)

        joint[:, 0] = text_tokens[:S]

        joint[:, 1] = moshi_audio_tokens[:, 0]

        for q in range(1, Q):
            for s in range(S):
                src_s = s - acoustic_delay
                if src_s >= 0:
                    joint[s, 1 + q] = moshi_audio_tokens[src_s, q]
                else:
                    joint[s, 1 + q] = pad_token

        return joint

    def build_joint_tokens_multi_stream(
        self,
        moshi_audio_tokens: np.ndarray,
        user_audio_tokens: np.ndarray,
        text_tokens: np.ndarray,
        acoustic_delay: int,
        pad_token: int = 0,
    ) -> np.ndarray:
        S = moshi_audio_tokens.shape[0]
        Q = self.ms_config.num_codebooks
        K_total = self.ms_config.total_depth_streams

        joint = np.zeros((S, K_total), dtype=np.int32)

        joint[:, 0] = text_tokens[:S]

        for s in range(S):
            joint[s, 1] = moshi_audio_tokens[s, 0]
            for q in range(1, Q):
                src_s = s - acoustic_delay
                if src_s >= 0:
                    joint[s, 1 + q] = moshi_audio_tokens[src_s, q]

        for s in range(S):
            joint[s, 1 + Q] = user_audio_tokens[s, 0]
            for q in range(1, Q):
                src_s = s - acoustic_delay
                if src_s >= 0:
                    joint[s, 1 + Q + q] = user_audio_tokens[src_s, q]

        return joint

    def compute_loss_mask(
        self,
        joint_tokens: np.ndarray,
        padding_weight: float = 0.5,
        semantic_weight: float = 100.0,
        acoustic_weight: float = 1.0,
        text_weight: float = 1.0,
        is_multi_stream: bool = True,
    ) -> np.ndarray:
        S, K = joint_tokens.shape
        weights = np.zeros((S, K), dtype=np.float32)

        pad_id = self.im_config.pad_token_id
        text_is_pad = joint_tokens[:, 0] == pad_id

        weights[:, 0] = np.where(text_is_pad, padding_weight, text_weight)

        weights[:, 1] = semantic_weight

        for q in range(1, self.ms_config.num_codebooks):
            weights[:, 1 + q] = acoustic_weight

        if is_multi_stream:
            weights[:, 1 + self.ms_config.num_codebooks] = 0.0

            for q in range(1, self.ms_config.num_codebooks):
                weights[:, 1 + self.ms_config.num_codebooks + q] = 0.0

        return weights

    def extract_inference_tokens(
        self,
        joint_tokens: np.ndarray,
        moshi_text: np.ndarray,
        moshi_audio: np.ndarray,
        user_audio_from_mic: np.ndarray,
    ) -> np.ndarray:
        Q = self.ms_config.num_codebooks
        K_total = self.ms_config.total_depth_streams
        S = moshi_audio.shape[0]

        joint = np.zeros((S, K_total), dtype=np.int32)
        joint[:, 0] = moshi_text
        joint[:, 1:1 + Q] = moshi_audio
        joint[:, 1 + Q:1 + 2 * Q] = user_audio_from_mic

        return joint

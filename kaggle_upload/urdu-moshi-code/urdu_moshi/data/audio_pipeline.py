from typing import Dict, Generator, Iterator, List, Optional, Tuple
import os
import random
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from model.inner_monologue import MultiStreamProcessor
from configs.model_config import MultiStreamConfig, InnerMonologueConfig


SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAMES_PER_SECOND = 12.5
SEGMENT_DURATION_SECONDS = 300.0
SEGMENT_FRAMES = int(SEGMENT_DURATION_SECONDS * FRAME_RATE)


def load_and_resample_audio(
    file_path: str,
    target_sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sample_rate, mono=True)
    except ImportError:
        try:
            import soundfile as sf
            audio, sr = sf.read(file_path, always_2d=False)
            if sr != target_sample_rate:
                import resampy
                audio = resampy.resample(audio, sr, target_sample_rate)
        except ImportError:
            raise ImportError("Install librosa or soundfile+resampy: pip install librosa")
    return audio.astype(np.float32)


def encode_audio_with_mimi(
    audio: np.ndarray,
    mimi_encode_fn,
) -> np.ndarray:
    import jax.numpy as jnp
    audio_jax = jnp.array(audio[None, None, :])
    tokens = mimi_encode_fn(audio_jax)
    return np.array(tokens[0])


class Stage1Dataset:
    def __init__(
        self,
        audio_dir: str,
        mimi_encode_fn,
        tokenizer,
        whisper_transcribe_fn,
        ms_config: MultiStreamConfig,
        im_config: InnerMonologueConfig,
        segment_duration: float = SEGMENT_DURATION_SECONDS,
        text_mask_prob: float = 0.3,
        text_delay_min: float = -0.6,
        text_delay_max: float = 0.6,
        acoustic_delay: int = 2,
        batch_size: int = 8,
        shuffle_buffer: int = 1000,
        num_parallel_calls: int = 8,
    ):
        self.audio_dir = audio_dir
        self.mimi_encode_fn = mimi_encode_fn
        self.tokenizer = tokenizer
        self.whisper_transcribe_fn = whisper_transcribe_fn
        self.ms_processor = MultiStreamProcessor(ms_config, im_config)
        self.im_config = im_config
        self.segment_frames = int(segment_duration * FRAME_RATE)
        self.text_mask_prob = text_mask_prob
        self.text_delay_range = (text_delay_min, text_delay_max)
        self.acoustic_delay = acoustic_delay
        self.batch_size = batch_size
        self.rng = np.random.default_rng(42)

    def _process_audio_file(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        try:
            audio = load_and_resample_audio(file_path)
        except Exception:
            return None

        min_samples = int(self.segment_frames / FRAME_RATE * SAMPLE_RATE)
        if len(audio) < min_samples:
            return None

        audio_tokens = encode_audio_with_mimi(audio[:min_samples], self.mimi_encode_fn)
        S, Q = audio_tokens.shape
        S = min(S, self.segment_frames)
        audio_tokens = audio_tokens[:S]

        text_delay = self.rng.uniform(*self.text_delay_range)
        transcript_result = self.whisper_transcribe_fn(file_path)

        if transcript_result is not None:
            words = transcript_result.get("words", [])
            word_texts = [w["word"] for w in words]
            word_starts = [w["start"] for w in words]
            text_tokens = build_text_tokens_with_timing(
                word_texts, word_starts, S, self.tokenizer, self.im_config, text_delay
            )
        else:
            text_tokens = np.full(S, self.im_config.pad_token_id, dtype=np.int32)

        if self.rng.random() < self.text_mask_prob:
            text_tokens = np.full(S, self.im_config.pad_token_id, dtype=np.int32)

        user_tokens = np.zeros_like(audio_tokens)

        joint = self.ms_processor.build_joint_tokens_single_stream(
            moshi_audio_tokens=audio_tokens,
            text_tokens=text_tokens,
            acoustic_delay=self.acoustic_delay,
        )

        return {"joint_tokens": joint}

    def iterate(self) -> Generator[Dict[str, np.ndarray], None, None]:
        audio_files = _find_audio_files(self.audio_dir)
        self.rng.shuffle(audio_files)

        batch_tokens = []
        for file_path in audio_files:
            sample = self._process_audio_file(file_path)
            if sample is None:
                continue
            batch_tokens.append(sample["joint_tokens"])
            if len(batch_tokens) == self.batch_size:
                yield {"joint_tokens": np.stack(batch_tokens, axis=0)}
                batch_tokens = []


class Stage2Dataset:
    def __init__(
        self,
        audio_dir: str,
        mimi_encode_fn,
        tokenizer,
        whisper_transcribe_fn,
        diarize_fn,
        ms_config: MultiStreamConfig,
        im_config: InnerMonologueConfig,
        segment_duration: float = SEGMENT_DURATION_SECONDS,
        acoustic_delay: int = 2,
        batch_size: int = 4,
    ):
        self.audio_dir = audio_dir
        self.mimi_encode_fn = mimi_encode_fn
        self.tokenizer = tokenizer
        self.whisper_transcribe_fn = whisper_transcribe_fn
        self.diarize_fn = diarize_fn
        self.ms_processor = MultiStreamProcessor(ms_config, im_config)
        self.im_config = im_config
        self.segment_frames = int(segment_duration * FRAME_RATE)
        self.acoustic_delay = acoustic_delay
        self.batch_size = batch_size
        self.rng = np.random.default_rng(0)

    def _process_diarized_file(
        self, file_path: str
    ) -> Optional[Dict[str, np.ndarray]]:
        try:
            audio = load_and_resample_audio(file_path)
            diarization = self.diarize_fn(file_path)
        except Exception:
            return None

        speaker_ids = list(set(seg["speaker"] for seg in diarization))
        if len(speaker_ids) < 2:
            return None

        main_speaker = self.rng.choice(speaker_ids)
        other_speakers = [s for s in speaker_ids if s != main_speaker]

        min_samples = int(self.segment_frames / FRAME_RATE * SAMPLE_RATE)
        audio = audio[:min_samples]

        main_audio = np.zeros_like(audio)
        other_audio = np.zeros_like(audio)

        for seg in diarization:
            start = int(seg["start"] * SAMPLE_RATE)
            end = int(seg["end"] * SAMPLE_RATE)
            start = min(start, len(audio))
            end = min(end, len(audio))
            if seg["speaker"] == main_speaker:
                main_audio[start:end] = audio[start:end]
            else:
                other_audio[start:end] = audio[start:end]

        main_tokens = encode_audio_with_mimi(main_audio, self.mimi_encode_fn)[:self.segment_frames]
        other_tokens = encode_audio_with_mimi(other_audio, self.mimi_encode_fn)[:self.segment_frames]

        transcript = self.whisper_transcribe_fn(file_path, speaker=main_speaker)
        if transcript:
            text_tokens = build_text_tokens_with_timing(
                [w["word"] for w in transcript.get("words", [])],
                [w["start"] for w in transcript.get("words", [])],
                self.segment_frames,
                self.tokenizer,
                self.im_config,
                text_delay=0.0,
            )
        else:
            text_tokens = np.full(self.segment_frames, self.im_config.pad_token_id, dtype=np.int32)

        joint = self.ms_processor.build_joint_tokens_multi_stream(
            moshi_audio_tokens=main_tokens,
            user_audio_tokens=other_tokens,
            text_tokens=text_tokens,
            acoustic_delay=self.acoustic_delay,
        )

        return {"joint_tokens": joint}

    def iterate(self) -> Generator[Dict[str, np.ndarray], None, None]:
        audio_files = _find_audio_files(self.audio_dir)
        self.rng.shuffle(audio_files)
        batch = []
        for f in audio_files:
            sample = self._process_diarized_file(f)
            if sample is None:
                continue
            batch.append(sample["joint_tokens"])
            if len(batch) == self.batch_size:
                yield {"joint_tokens": np.stack(batch, axis=0)}
                batch = []


def build_text_tokens_with_timing(
    words: List[str],
    word_starts: List[float],
    total_frames: int,
    tokenizer,
    im_config: InnerMonologueConfig,
    text_delay: float = 0.0,
) -> np.ndarray:
    from model.inner_monologue import align_text_to_frames
    word_token_lists = tokenizer.tokenize_word_list(words) if words else []
    adjusted_starts = [t + text_delay for t in word_starts]
    return align_text_to_frames(
        word_token_lists,
        adjusted_starts,
        total_frames,
        im_config,
    )


def _find_audio_files(audio_dir: str) -> List[str]:
    audio_extensions = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
    files = []
    for root, _, filenames in os.walk(audio_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in audio_extensions:
                files.append(os.path.join(root, fname))
    return files


def apply_stage4_augmentation(
    user_audio: np.ndarray,
    moshi_audio: np.ndarray,
    config,
    rng: np.random.Generator,
) -> np.ndarray:
    augmented = user_audio.copy()

    if rng.random() < config.user_stream_gain_prob:
        gain_db = rng.uniform(config.user_stream_gain_min_db, config.user_stream_gain_max_db)
        gain_linear = 10 ** (gain_db / 20.0)
        augmented = augmented * gain_linear

    if rng.random() < config.noise_add_prob:
        noise_level_db = rng.uniform(config.noise_snr_min_db, config.noise_snr_max_db)
        noise_power = np.mean(augmented ** 2) / (10 ** (noise_level_db / 10.0)) + 1e-9
        noise = rng.normal(0, np.sqrt(noise_power), augmented.shape).astype(np.float32)
        augmented = augmented + noise

    if rng.random() < config.echo_reverb_prob:
        echo_scale = rng.uniform(config.echo_scale_min, config.echo_scale_max)
        delay_samples = int(rng.uniform(config.echo_delay_min_ms, config.echo_delay_max_ms) / 1000.0 * SAMPLE_RATE)
        echo = np.zeros_like(augmented)
        if delay_samples < len(moshi_audio):
            echo_len = min(len(augmented), len(moshi_audio) - delay_samples)
            echo[delay_samples:delay_samples + echo_len] = moshi_audio[:echo_len] * echo_scale
        augmented = augmented + echo

    augmented = np.clip(augmented, -1.0, 1.0)
    return augmented

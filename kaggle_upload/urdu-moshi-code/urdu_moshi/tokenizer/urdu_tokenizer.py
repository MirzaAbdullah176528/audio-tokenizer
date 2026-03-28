import os
import re
from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm
import numpy as np


PAD_TOKEN = "<pad>"
EPAD_TOKEN = "<epad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, EPAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_ID = 0
EPAD_ID = 1
BOS_ID = 2
EOS_ID = 3
UNK_ID = 4


URDU_UNICODE_RANGES = [
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
    (0x0020, 0x0020),
    (0x0030, 0x0039),
    (0x0660, 0x0669),
]


def normalize_urdu_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\u0020-\u0039\u0660-\u0669\u06F0-\u06F9،؟!؛\.\,\-]', ' ', text)
    text = text.replace('\u0640', '')
    text = text.replace('\u200c', ' ')
    text = text.replace('\u200d', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_urdu_tokenizer(
    corpus_files: List[str],
    output_prefix: str,
    vocab_size: int = 32000,
    character_coverage: float = 0.9999,
    model_type: str = "unigram",
    split_digits: bool = True,
    byte_fallback: bool = True,
    num_threads: int = 16,
    input_sentence_size: int = 10_000_000,
    shuffle_input_sentence: bool = True,
    pad_id: int = PAD_ID,
    unk_id: int = UNK_ID,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
) -> None:
    corpus_str = ",".join(corpus_files)
    user_defined_symbols = [EPAD_TOKEN]

    spm.SentencePieceTrainer.train(
        input=corpus_str,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        split_digits=split_digits,
        byte_fallback=byte_fallback,
        num_threads=num_threads,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle_input_sentence,
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
        user_defined_symbols=",".join(user_defined_symbols),
        normalization_rule_name="nmt_nfkc_cf",
        add_dummy_prefix=False,
        remove_extra_whitespaces=True,
    )
    print(f"Tokenizer saved to {output_prefix}.model and {output_prefix}.vocab")


class UrduTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        self.vocab_size = self.sp.get_piece_size()
        self.pad_id = PAD_ID
        self.epad_id = EPAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        normalize: bool = True,
    ) -> List[int]:
        if normalize:
            text = normalize_urdu_text(text)
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        filtered = [i for i in ids if i not in (self.pad_id, self.epad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(filtered)

    def encode_with_timestamps(
        self,
        word_tokens: List[List[int]],
        word_start_frames: List[int],
        total_frames: int,
        frame_rate: float = 12.5,
    ) -> np.ndarray:
        sequence = np.full(total_frames, self.pad_id, dtype=np.int32)
        for i, (tokens, start_frame) in enumerate(zip(word_tokens, word_start_frames)):
            if start_frame > 0:
                sequence[start_frame - 1] = self.epad_id
            for j, tok in enumerate(tokens):
                pos = start_frame + j
                if pos < total_frames:
                    sequence[pos] = tok
        return sequence

    def tokenize_word_list(self, words: List[str]) -> List[List[int]]:
        return [self.sp.encode(w, out_type=int) for w in words]

    def id_to_piece(self, id: int) -> str:
        return self.sp.id_to_piece(id)

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)


def build_urdu_tokenizer_from_whisper_transcripts(
    transcript_dir: str,
    output_prefix: str,
    vocab_size: int = 32000,
) -> UrduTokenizer:
    import tempfile
    import glob

    transcript_files = glob.glob(os.path.join(transcript_dir, "**/*.txt"), recursive=True)
    if not transcript_files:
        raise ValueError(f"No .txt files found in {transcript_dir}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        temp_path = f.name
        for fpath in transcript_files:
            with open(fpath, "r", encoding="utf-8") as src:
                for line in src:
                    normalized = normalize_urdu_text(line.strip())
                    if normalized:
                        f.write(normalized + "\n")

    train_urdu_tokenizer(
        corpus_files=[temp_path],
        output_prefix=output_prefix,
        vocab_size=vocab_size,
    )
    os.unlink(temp_path)

    return UrduTokenizer(f"{output_prefix}.model")

"""Tiện ích trích xuất đặc trưng Wav2Vec2-base cho project ResLSTM-SER.

Thay thế cho `librosa.feature.mfcc` / `chroma_stft`. Chỉ chứa đúng những
gì cần để `train.ipynb` gọi vào: load processor/model một lần và trích
`last_hidden_state` (T, 768) cho một file wav.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def load_wav2vec2(model_name: str = "facebook/wav2vec2-base", device: str = "cpu"):
    """Load Wav2Vec2Processor + Wav2Vec2Model một lần, dùng lại cho mọi file.

    Trả về (processor, model). Model được set `.eval()` vì ta chỉ dùng
    Wav2Vec2 như feature extractor cố định (không fine-tune) trong pipeline
    trích xuất offline này.
    """
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def extract_wav2vec2_features(
    file_path: str,
    processor,
    model,
    device: str = "cpu",
    target_sample_rate: int = 16_000,
) -> Optional[np.ndarray]:
    """Trích `last_hidden_state` (time_len, 768) cho một file wav.

    Args:
        file_path: đường dẫn tới file .wav.
        processor: `Wav2Vec2Processor` đã load qua `load_wav2vec2`.
        model: `Wav2Vec2Model` đã load qua `load_wav2vec2`.
        device: 'cpu' hoặc 'cuda'.
        target_sample_rate: Wav2Vec2-base yêu cầu 16kHz.

    Returns:
        features: (time_len, 768) numpy array, hoặc None nếu lỗi.
    """
    import librosa
    import torch

    try:
        audio, _ = librosa.load(file_path, sr=target_sample_rate, mono=True)

        inputs = processor(
            audio, sampling_rate=target_sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            # outputs = model(input_values=input_values)
            outputs = model(
                input_values=input_values,
                output_hidden_states=True
)

        # (1, T, 768) -> (T, 768)
        # Lấy output của Transformer Layer 8
        # hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        hidden_states = outputs.hidden_states[8].squeeze(0).cpu().numpy()
        return hidden_states.astype(np.float32)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

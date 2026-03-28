from setuptools import setup, find_packages

setup(
    name="urdu-moshi",
    version="0.1.0",
    description="Urdu Full-Duplex Speech Dialogue System based on Moshi, built in JAX/Flax for TPU v5e-8",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "jax[tpu]>=0.4.25",
        "flax>=0.8.2",
        "optax>=0.2.2",
        "orbax-checkpoint>=0.5.10",
        "transformers>=4.40.0",
        "sentencepiece>=0.2.0",
        "numpy>=1.26.0",
        "torch>=2.2.0",
    ],
    extras_require={
        "audio": [
            "librosa>=0.10.1",
            "soundfile>=0.12.1",
            "resampy>=0.4.2",
            "whisper-timestamped>=1.15.4",
        ],
        "diarization": [
            "pyannote.audio>=3.1.1",
        ],
        "dev": [
            "pytest>=8.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
        ],
    },
)

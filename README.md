# ResNet18 and GPT-2 Quantization for Efficient AI Inference

This repository documents a PyTorch quantization study across both convolutional neural networks and transformer-based language models. The project explores post-training quantization, clipping-factor selection, straight-through-estimator-based quantization-aware training, and quantization granularity choices for ResNet-18 on CIFAR-10 and GPT-2 on LAMBADA.

## Overview

The notebook is organized into five parts:

1. **Fixed-Point Post-Training Quantization**
   - Implements tensor-wise asymmetric quantization.
   - Implements filter-wise asymmetric quantization for CNN weights.
   - Compares the accuracy impact of tensor-wise vs. filter-wise quantization.

2. **Floating-Point Representation**
   - Decomposes FP32 values into sign, exponent, and mantissa components.
   - Demonstrates how floating-point numbers are represented internally.

3. **Symmetric Quantization and Clipping-Factor Selection**
   - Implements tensor-wise symmetric quantization.
   - Sweeps clipping factors for 4-bit weight quantization.
   - Studies how clipping range affects quantization error and model accuracy.

4. **Quantization-Aware Training**
   - Implements the Straight Through Estimator (STE) for quantization.
   - Applies 4-bit weight and activation quantization during training.
   - Fine-tunes the quantized model to recover accuracy.

5. **Transformer Language Model Quantization**
   - Loads GPT-2-small using a nanoGPT-style implementation.
   - Evaluates GPT-2 on the LAMBADA benchmark.
   - Implements tensor-wise, channel-wise, and token-wise quantization for transformer linear layers.
   - Compares 4-bit, 8-bit, and 16-bit quantization settings.

## Model and Datasets

| Component | Setup |
|---|---|
| CNN model | ResNet-18 |
| CNN dataset | CIFAR-10 |
| Language model | GPT-2 small |
| Language-model benchmark | LAMBADA |
| Framework | PyTorch |
| Profiling / utilities | THOP, TorchVision, Hugging Face Datasets, tiktoken |
| Hardware | CUDA-enabled GPU |

## Key Results

### ResNet-18 Quantization on CIFAR-10

| Experiment | Result |
|---|---:|
| Full-precision baseline accuracy | 93.79% |
| Tensor-wise PTQ, 8-bit activations / 8-bit weights | 93.66% |
| Filter-wise PTQ, 8-bit activations / 8-bit weights | 93.80% |
| Tensor-wise PTQ, 4-bit activations / 4-bit weights | 85.42% |
| Filter-wise PTQ, 4-bit activations / 4-bit weights | 91.32% |
| Best 4-bit symmetric clipping factor | 0.75 |
| Best 4-bit clipping-factor accuracy | 91.97% |
| Initial 4-bit QAT model accuracy | 72.95% |
| Final 4-bit QAT model accuracy | 91.28% |

### GPT-2 Quantization on LAMBADA

| Experiment | LAMBADA Accuracy |
|---|---:|
| Full-precision GPT-2 baseline | 0.4657 |
| Tensor-wise 8-bit activations / 8-bit weights | 0.3840 |
| Tensor-wise 16-bit activations / 16-bit weights | 0.4659 |
| Channel/token-wise 8-bit activations / 8-bit weights | 0.4685 |
| Channel/token-wise 16-bit activations / 16-bit weights | 0.4657 |

## Main Takeaways

Filter-wise quantization performs better than tensor-wise quantization for CNN weights because each filter receives its own quantization scale. This reduces rounding error when different filters have different numerical ranges.

For 4-bit ResNet-18 quantization, clipping the weight range can improve accuracy. A clipping factor below 1.0 sacrifices a few outlier weights but gives finer resolution to the majority of values. In this experiment, a clipping factor of 0.75 produced the best result.

Quantization-aware training substantially improves low-bit performance. The 4-bit quantized ResNet-18 model initially dropped to 72.95% accuracy, but STE-based QAT recovered it to 91.28%.

For GPT-2, 4-bit quantization is too aggressive under the simple post-training setup used here. 8-bit quantization is the best practical sweet spot: it gives much stronger compression than 16-bit while preserving accuracy when channel-wise weight quantization and token-wise activation quantization are used.

## Technologies

Python, PyTorch, TorchVision, CUDA, THOP, NumPy, Matplotlib, tqdm, Hugging Face Datasets, tiktoken, GPT-2, CIFAR-10, LAMBADA

## Possible Extensions

- Measure real inference latency and memory usage for each quantization setting.
- Compare PTQ and QAT on MobileNet, ResNet-34, or Vision Transformers.
- Implement per-channel activation quantization for CNNs.
- Add SmoothQuant-style scaling for GPT-2.
- Export quantized models to ONNX or TorchScript.
- Compare simple PTQ against PyTorch native quantization tooling.

## Repository Structure

```text
.
├── README.md
└── resnet18_gpt2_quantization_efficient_ai.ipynb
```

## Notes

This repository is intended as a learning-focused quantization study. It emphasizes implementation, accuracy tradeoffs, and intuition behind quantization granularity rather than production-grade quantized deployment.

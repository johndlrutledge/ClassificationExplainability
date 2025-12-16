# AI Model Explainability - Interactive Demo

A hands-on demonstration of modern AI explainability techniques for computer vision models. This project showcases Grad-CAM, Integrated Gradients, and SmoothGrad through an interactive web interface built with Gradio.

## 🎯 Overview

This is a simple ML test designed to help you understand how different explainability techniques work. Upload an image, select a model, and see which pixels the model focuses on when making predictions. Perfect for learning, debugging, and exploring model behavior.

## ✨ Features

- **Interactive Web Interface**: Simple Gradio UI for real-time exploration
- **Multiple Explainability Methods**:
  - **Grad-CAM**: See which image regions activate each neuron
  - **Integrated Gradients**: Understand how each pixel contributes to predictions
  - **SmoothGrad**: Reduce noise in gradient-based explanations
- **12 Pre-trained Models**: Test with ResNet, Vision Transformer (ViT), ConvNeXt, and more
- **Top-5 Predictions**: See model confidence across multiple classes
- **Example Gallery**: Pre-loaded images for quick testing and comparison

## 🚀 Supported Models (most that support AutoModelForImageClassification)

1. **nvidia/mit-b0** - Efficient Vision Transformer
2. **facebook/convnext-base-224** - Modern ConvNeXt architecture
3. **microsoft/resnet-18** - Lightweight ResNet
4. **microsoft/resnet-50** - Standard ResNet (default)
5. **Falconsai/nsfw_image_detection** - Content safety classifier
6. **microsoft/swin-tiny-patch4-window7-224** - Hierarchical Vision Transformer
7. **google/vit-base-patch16-224** - Base Vision Transformer
8. **juppy44/plant-identification-2m-vit-b** - Plant species classifier
9. **BinhQuocNguyen/food-recognition-model** - Food item classifier
10. **LucyintheSky/pose-estimation-front-side-back** - Pose estimation model
11. **google/derm-foundation** - Dermatology analysis model
12. **timm/csatv2** - General-purpose vision model

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 4GB+ RAM (sufficient for testing)

### Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/johndlrutledge/ClassificationExplainability.git
    cd ClassificationExplainability
    ```

2. **Create a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**:

    ```bash
    python app.py
    ```

5. **Access the interface**: Open your browser to `http://localhost:7860`

## 📋 Requirements

```bash
torch
numpy
matplotlib
pillow
gradio
saliency
transformers
```

## 🔧 Configuration

### Changing the Default Model

Edit `app.py` and modify the `model_choice` variable:

```python
model_choice = 3  # Change index (0-11) to select different model
```

### Adding Custom Models

To add your own model:

1. Add the model identifier to the `model_names` list
2. Ensure the model follows the Hugging Face `AutoModelForImageClassification` interface
3. Update the model configuration in the `Model` class if needed


## 🚀 Quick Start

### Run Locally

```bash
python app.py
```

The app will be available at `http://localhost:7860`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Contribution

- Additional explainability methods
- Support for more model architectures
- Enhanced visualization options
- Batch processing capabilities

## 📚 References

### Core Libraries

- **[Saliency](https://github.com/PAIR-code/saliency)**: Core explainability implementations
- **[Gradio](https://gradio.app/)**: Interactive web interface
- **[Transformers](https://huggingface.co/docs/transformers)**: Model loading and preprocessing
- **[PyTorch](https://pytorch.org/)**: Deep learning framework
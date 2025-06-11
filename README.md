# PictSure: In-Context Learning for Image Classification

PictSure is a deep learning library designed for **in-context learning** using images and labels. It allows users to provide a set of labeled reference images and then predict labels for new images based on those references. This approach eliminates the need for traditional training, making it highly adaptable for various classification tasks.

<p align="center">
  <img src="images/Flow-Chart.png" alt="The classification process" width="90%" />
</p>

## Features
- **In-Context Learning**: Predict labels for new images using a set of reference images without traditional model training.
- **Multiple Model Architectures**: Choose between ResNet and ViT-based models for your specific needs.
- **Pretrained Models**: Use our pretrained models or train your own.
- **Torch Compatibility**: Fully integrated with PyTorch, supporting CPU and GPU.
- **Easy-to-use CLI**: Manage models and weights through a simple command-line interface.

## Installation
1. Clone this repository
```bash
git clone https://git.ni.dfki.de/pictsure/pictsure-library
```
2. Navigate into the folder
```bash
cd pictsure-library
```
3. Install the pip package
```bash
pip install .
```

## Quick Start
```python
from PictSure import PictSure
import torch

# Initialize the model (using ViT as an example)
model = PictSure(
    embedding='vit',  # or 'resnet'
    pretrained=True,  # use pretrained weights
    device='cuda'     # or 'cpu'
)

# Set your reference images and labels
model.set_context_images(reference_images, reference_labels)

# Make predictions on new images
predictions = model.predict(new_images)
```

## Command Line Interface
PictSure comes with a command-line interface to manage models and weights:

### List Available Models
```bash
pictsure list-models
```
This command shows all available models, their status (downloaded/not downloaded), and detailed information about each model.

### Remove Model Weights
```bash
pictsure remove <model_name> [--force]
```
Remove the weights of a specific model. Available models are:
- `ViTPreAll`: ViT-based model
- `ResPreAll`: ResNet-based model

Use the `--force` or `-f` flag to skip the confirmation prompt.

## Examples
For a complete working example, check out the Jupyter notebook in the Examples directory:
```bash
Examples/example.ipynb
```
This notebook demonstrates:
- Model initialization
- Loading and preprocessing images
- Setting up reference images
- Making predictions
- Visualizing results

## License
This project is open-source under the MIT License.

## Contributing
Contributions and suggestions are welcome! Open an issue or submit a pull request.

## Contact
For questions or support, open an issue on GitHub.


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
```bash
pip install PictSure
```

## Quick Start
```python
from PictSure import PictSure
import torch

DEVICE = "cpu" # or cuda, mps

model = PictSure.from_pretrained("pictsure/pictsure-vit")
model = model.to(DEVICE)

# Set your reference images and labels
model.set_context_images(reference_images, reference_labels)

# Make predictions on new images
predictions = model.predict(new_images)
```

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

## Citation

If you use this work, please cite it using the following BibTeX entry:

```bibtex
@article{schiesser2025pictsure,
  title={PictSure: Pretraining Embeddings Matters for In-Context Learning Image Classifiers},
  author={Schiesser, Lukas and Wolff, Cornelius and Haas, Sophie and Pukrop, Simon},
  journal={arXiv preprint arXiv:2506.14842},
  year={2025}
}
```

## License
This project is open-source under the MIT License.

## Contributing
Contributions and suggestions are welcome! Open an issue or submit a pull request.

## Contact
For questions or support, open an issue on GitHub.


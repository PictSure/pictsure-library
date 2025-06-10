"""
Configuration file for PictSure containing model information and pretrained weights URLs.
"""

PRETRAINED = {
    'resnet': {
        'url': 'https://cloud.dfki.de/owncloud/index.php/s/AXTx2MgLyyMWKWg/download/best_loss_model.pt',
        'embed_model': 'resnet18',
        'num_classes': 10,
        'nheads': 8,
        'nlayer': 4,
        'name': "ResPreAll",
        'resolution': (224, 224),
        'size': 53,
    },
    'vit': {
        'url': 'https://cloud.dfki.de/owncloud/index.php/s/xRZGGDtep9rfYGQ/download/best_acc_model.pt',
        'embed_model': 'vit',
        'num_classes': 10,
        'nheads': 8,
        'nlayer': 4,
        'name': "ViTPreAll",
        'resolution': (224, 224),
        'size': 128,
    }
} 
"""深度学习模型模块"""

from .spectrum_classifier import (
    Spec2ClassCNN,
    Spec2ClassTransformer,
    Spec2ClassHybrid,
    Spec2ClassResNet,
    create_model
)

from .bert_classifier import (
    BertSpectrumClassifier,
    create_bert_classifier
)

__all__ = [
    'Spec2ClassCNN',
    'Spec2ClassTransformer',
    'Spec2ClassHybrid',
    'Spec2ClassResNet',
    'create_model',
    'BertSpectrumClassifier',
    'create_bert_classifier'
]

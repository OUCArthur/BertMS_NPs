# BertMS_NPs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24389538.svg)](https://doi.org/10.5281/zenodo.24389538)

A BERT-based approach for rapid dereplication of natural products from mass
spectrometry (MS/MS) data: spectral embedding via a from-scratch BERT
implementation, similarity-network construction, clustering, and
structure/database annotation.

## Project layout

```
model/              From-scratch BERT implementation (architecture + pretraining heads)
utils/              Dataset loading, BERT/DL inference helpers, attention-extraction tools
models/             Deep-learning baseline classifiers (CNN / Transformer / Hybrid / ResNet)
matchms_ws/         Vendored & adapted copy of matchms (spectral I/O, filtering, similarity)
bertms_ws/          Spectrum-to-BERT-vector utilities (Spec2Vec-style interface)
bertms_analysis/    Main analysis package: preprocessing, network construction, annotation
Tasks/              BertMS pretraining script
document/           Evaluation scripts used in the manuscript (AUC/AUPR, ranking fairness, ...)
ablation_embedding/ Embedding ablation study
注意力/              Attention-weight extraction and visualization
run_analysis_refactored.ipynb   Main entry-point notebook for the analysis pipeline
```

## Installation

```bash
pip install -r requirements.txt
```

`matchms_ws` is an adapted copy of the [matchms](https://github.com/matchms/matchms)
project (Apache-2.0); the original `matchms` package is also required (see
`requirements.txt`) for a few type references.

## Data and pretrained weights

Raw spectra, the pretrained `bert_base_ms` checkpoint, and training logs are
**not** included in this repository due to size. See [TODO: add download link /
instructions] for how to obtain them. Point `config.pretrained_model_dir` /
`config.dataset_dir` in `run_analysis_refactored.ipynb` at wherever you place
them.

## Usage

Open and run `run_analysis_refactored.ipynb`. It walks through:

1. Loading and preprocessing MSP/MGF spectra
2. Loading the pretrained BertMS model and computing pairwise spectral similarity
3. Building a molecular similarity network and clustering it
4. Enhanced annotation (database search + structural-class prediction)
5. Constrained clustering (size / m/z constraints)

To pretrain BertMS from scratch, see `Tasks/TasksForPretraining_ms.py`.

## Acknowledgements

- [matchms](https://github.com/matchms/matchms) — spectral data handling (Apache-2.0)
- The core BERT implementation is adapted from the
  [BertWithPretrained](https://github.com/moon-hotel/BertWithPretrained) tutorial project.

## How to cite

If you use this code, please cite the archived release:

```
zln. (2026). OUCArthur/BertMS_NPs: v1.0.0 - Initial public release.
Zenodo. https://doi.org/10.5281/zenodo.24389538
```

## License

MIT — see [LICENSE](LICENSE).

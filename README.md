# FlowTP

This repository contains the official implementation and manuscript for our paper **FlowTP: Conditional Flow Matching with Dual-Schedule Guidance for Controllable Therapeutic Peptide Generation**, a conditional generative framework for therapeutic peptide design.

---

## 1. Repository Structure

```text
.
|-- 代码/    # Source code, datasets, configuration, and local ESM-2 weights
`-- 论文/    # Manuscript source and figures
```

All commands below should be run from the `代码` directory:

```bash
cd 代码
```

---

## 2. Environment Setup

We recommend using **conda** to reproduce the experimental environment:

```bash
conda env create -f environment.yml
conda activate FlowTP
```

The repository includes the local `ESM2-8M` model files used by the code. If you encounter package conflicts, please ensure that your conda version is up to date and that your CUDA installation is compatible with the PyTorch version specified in `environment.yml`.

---

## 3. Model Training

To train the conditional flow-matching model from scratch, run:

```bash
python train.py
```

The training configuration is defined in `config.ini`. Training and validation data are read from `data/split_v2/`, and checkpoints are written to `save_model/`. If `save_model/checkpoint_flow.pth` exists, training resumes automatically from that checkpoint.

### Notation Convention

The implementation uses the endpoint notation in the reverse direction from the paper: in the paper, $x_0$ denotes the data latent and $x_1$ denotes Gaussian noise, whereas the code uses `x0` for Gaussian noise and `x1` for the data latent. Accordingly, the code advances time from noise to data, with $t_{\mathrm{code}} = 1 - t_{\mathrm{paper}}$. The interpolation path, target velocity, training-time input, and sampling updates all follow this convention consistently, so this is only a difference in notation and time parameterization rather than a difference in the underlying method.

---

## 4. Sampling and Metric Evaluation

After preparing the trained flow-matcher and task-specific decoder checkpoints in `save_model/`, generate peptide sequences and compute the built-in evaluation metrics with:

```bash
# Antimicrobial peptides (AMPs)
python sample.py --task amp --count 1000 --output results/amp.fasta

# Antifungal peptides (AFPs)
python sample.py --task afp --count 1000 --output results/afp.fasta

# Antiviral peptides (AVPs)
python sample.py --task avp --count 1000 --output results/avp.fasta
```

Use `--generate-only` to save sequences without computing metrics:

```bash
python sample.py --task amp --count 1000 --output results/amp.fasta --generate-only
```

To evaluate an existing FASTA file using the same metric implementation, run:

```bash
python evaluate.py --task amp --input results/amp.fasta
```

The built-in evaluation reports pseudo-perplexity, sequence entropy, similarity to the corresponding reference set, and instability.

---

## 5. Activity Evaluation (External Predictors)

Functional activity is evaluated using **external, pre-trained web servers**, which act as fixed labeling oracles. Upload or submit the generated peptide sequences to the platform corresponding to the target therapeutic class:

- **Antimicrobial Activity (CAMP):** [https://camp.bicnirrh.res.in/predict/](https://camp.bicnirrh.res.in/predict/)
- **Antifungal Activity (AntifungiPept):** [https://antifungipept.chemoinfolab.com/design/predict-activity/](https://antifungipept.chemoinfolab.com/design/predict-activity/)
- **Antiviral Peptide Activity (AVP):** [http://121.36.197.223:8006/AVP](http://121.36.197.223:8006/AVP)

The **Activity Score** is defined as the proportion of generated sequences predicted to be active by the corresponding predictor. We use these external classifiers to evaluate the proportion of generated sequences exhibiting the specified therapeutic effect; the resulting proportions are collectively referred to as **Activity**. These scores are evaluation results and weak supervision signals rather than direct optimization targets during training.

> These external tools are not part of this repository and should be cited appropriately in academic use.

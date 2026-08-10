# FlowTP

This repository contains the official implementation for our paper **FlowTP: Flow-Matching for Controllable Therapeutic Peptide Generation**, a conditional generative framework for peptide design.

---

## 1. Environment Setup

We recommend using **conda** to reproduce the experimental environment.

```bash
conda env create -f environment.yml
conda activate FlowTP
```

> If you encounter package conflicts, please ensure your conda version is up to date.

---

## 2. Model Training

To train the model from scratch, simply run:

```bash
python train.py
```

### Notation Convention

The implementation uses the endpoint notation in the reverse direction from the paper: in the paper, $x_0$ denotes the data latent and $x_1$ denotes Gaussian noise, whereas the code uses `x0` for Gaussian noise and `x1` for the data latent. Accordingly, the code advances time from noise to data, with $t_{\mathrm{code}} = 1 - t_{\mathrm{paper}}$. The interpolation path, target velocity, training-time input, and sampling updates all follow this convention consistently, so this is only a difference in notation and time parameterization rather than a difference in the underlying method.

---

## 3. Sampling and Metric Evaluation

After training, you can sample new peptide sequences and compute partial evaluation metrics using:

```bash
python sample.py
```

---

## 4. Activity Evaluation (External Predictors)

Functional activity is evaluated using **external, pre-trained web servers**, which act as fixed labeling oracles.

Please upload or submit the generated peptide sequences to the following platforms:

* **Antimicrobial Activity (CAMP)**
  [https://camp.bicnirrh.res.in/predict/](https://camp.bicnirrh.res.in/predict/)

* **Antifungal Activity (AntifungiPept)**
  [https://antifungipept.chemoinfolab.com/design/predict-activity/](https://antifungipept.chemoinfolab.com/design/predict-activity/)

* **Antiviral Peptide Activity (AVP)**
  [http://121.36.197.223:8006/AVP](http://121.36.197.223:8006/AVP)

The Activity Score is defined as the proportion of generated sequences predicted to be active by the corresponding predictor. We use external classifiers to evaluate the proportion of generated sequences exhibiting the specified therapeutic effect; the resulting proportions are collectively referred to as Activity. These activity scores provide weak supervision signals rather than direct optimization targets during training.

> These tools are not part of this repository and should be cited appropriately in academic use.



---


## Contact

For questions or issues, please open an issue or contact the authors.

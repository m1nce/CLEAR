# Contextual Latent Embedding for Adaptive Representation

The **CLEAR** framework is designed to advance label-free concept bottleneck methods 
by developing a **contextually adaptive latent space** that enhances interpretability 
and robustness in complex, real-world settings. By using unsupertvissed learning techniques, 
CLEAR identifies latent pseudo-concepts within data, enabling the model to distinguish 
between genuine operational anomalies and context-induced variations (such as environmental changes 
or sensor noise). CLEAR's statistical foundations and context-awareness allow for a structured, 
interpretable bottleneck layer that makes data patterns more transparent and actionable, 
even in the absence of labeled data.

## Setup

1. Install Python (3.9) and PyTorch (1.13).
2. Install dependencies by running `pip install -r requirements.txt`
3. Download pretrained models by running  `bash download_models.sh` (they will be unpacked to `saved_models`)
4. Download and process CUB dataset by running `bash download_cub.sh` 
5. Download ResNet18(Places365) backbone by running `bash download_rn18_places.sh`

We do not provide download instructions for ImageNet data, to evaluate using your own copy of ImageNet you must set the correct path in `DATASET_ROOTS["imagenet_train"]` and `DATASET_ROOTS["imagenet_val"]` variables in `data_utils.py`.

## Running the models

### 1. Creating Concept Sets (Optional):
A. Create initial concept set using GPT-3 - `GPT_initial_concepts.ipynb`, do this for all 3 prompt types (can be skipped if using the concept sets we have provided). NOTE: This step costs money and you will have to provide your own `openai.api_key`.

B. Process and filter the conceptset by running `GPT_conceptset_processor.ipynb` (Alternatively get ConceptNet concepts by running ConceptNet_conceptset.ipynb)

### 2. Train LF-CBM

Train a concept bottleneck model on CIFAR10 by running:

`python train_cbm.py --concept_set data/concept_sets/cifar10_filtered.txt`


### 3. Evaluate trained models

Evaluate the trained models by running `evaluate_cbm.ipynb`. This measures model accuracy, creates barplots explaining individual decisions and prints final layer weights which are the basis for creating weight visualizations.

Additional evaluations and reproductions of our model editing experiments are available in the notebooks of `experiments` directory.

## Results

High Accuracy:

|                   |         |          | Dataset |           |          |
|-------------------|---------|----------|---------|-----------|----------|
| Model             | CIFAR10 | CIFAR100 | CUB200  | Places365 | ImageNet |
| Standard          | 88.80%  | 70.10%   | 76.70%  | 48.56%    | 76.13%   |
| Standard (sparse) | 82.96%  | 58.34%   | **75.96%**  | 38.46%    | **74.35%**   |
| Label-free CBM    | **86.37%** | **65.27%**   | 74.59%  | **43.71%**   | 71.98%   |

For commands to train Label-free CBM and Standard (sparse) models on all 5 datasets, see `training_commands.txt`.

Explainable Decsisions:

![](data/lf_cbm_ind_decision.png)

## Sources

CUB dataset: https://www.vision.caltech.edu/datasets/cub_200_2011/

Sparse final layer training: https://github.com/MadryLab/glm_saga

Explanation bar plots adapted from: https://github.com/slundberg/shap

CLIP: https://github.com/openai/CLIP

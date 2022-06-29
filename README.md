# Federated Biobank Project

## Motivation

## TG Environment Installation

It's recommended to work with TG codebase using `conda` environemnt.

1. Install `conda`: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html.

2. Activate `conda` environment suing requirenemnts file `requirements_tg.txt`:
```
conda create --name genx --file tg_requirements.txt
```

3. Activate the environment:
```
conda activate genx
```

4. Install `pgenlib` from PLINK's repo:
```
git clone https://github.com/chrchang/plink-ng.git
cd plink-ng/2.0/Python
python3 setup.py build_ext
pip install -e .
```

## Structure

 - **split** module generates node datasets from the whole UKB dataset based on self-reported ancestry.
 - **qc** module encapsulates node-based quality control.
 - **dimred** module performs different strategies of dimensionality reduction.
 - **fl** module compares various FL strategies on selected SNPs.

## Visualisation
### Dash App
Run dash_app.py and open the link that appears in console in a browser. There assign filter+value or graph elements (x-axis, y-axis, color, etc.) to columns via dropdowns. Then press submit.


# Federated Biobank Project

## Motivation

## Installation
 - For running TG code requirements_tg.txt can be used
 - Install pgenlib from PLINK's repo (change accordingly)
   - git clone https://github.com/chrchang/plink-ng.git
   - cd plink-ng/2.0/Python
   - /usr/local/bin/python setup.py build_ext
   - pip install -e .


## Structure

 - **split** module generates node datasets from the whole UKB dataset based on self-reported ancestry.
 - **qc** module encapsulates node-based quality control.
 - **dimred** module performs different strategies of dimensionality reduction. 
 - **fl** module compares various FL strategies on selected SNPs.

## Visualisation
### Dash App
Run dash_app.py and open the link that appears in console in a browser. There assign filter+value or graph elements (x-axis, y-axis, color, etc.) to columns via dropdowns. Then press submit.


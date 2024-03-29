# pandda_score

CNN(SqueezeNet)-based classifier to identify residues requiring remodelling in PanDDA event maps

# motivations
Traditional statistical metrics, such as RSCCs, tend to systematically give different scores in an occupancy/(1-BDC)-dependent manner. A trained neural network may be able to produce scores in an occupancy-independent manner.

# Dependencies
```shell
conda create -n pandda_env python=3.9.13 biopython=1.78 pandas=1.4.3 tqdm=4.64.0 matplotlib=2.2.3 numpy=1.23.1 seaborn=0.11.2
conda activate pandda_env
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge -y gemmi=0.4.5 torchinfo=1.7.0 plotly=5.10.0 fire=0.4.0 scikit-learn=1.1.1 pytorch-lightning
python -m pip install notebook
python -m pip install more-itertools
python -m pip install -e .
```
# splitting dataset into training and test sets
```shell
python learning/pull_data/split_dataset.py training/training_data.csv --test HERC2A,INPP5DA,NSP14,IMPA1A -s

```
# panddaproject

# dependencies for diamond clusters
```shell
conda create -n pandda_env python=3.9.13 biopython=1.78 pandas=1.4.3 tqdm=4.64.0 matplotlib=2.2.3 numpy=1.23.1
conda activate pandda_env
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge -y gemmi=0.4.5 torchinfo=1.7.0 plotly-5.10.0 fire=0.4.0
python -m pip install notebook
python -m pip install -e .
```

# splitting dataset into training and test sets
```shell
python learning/pull_data/split_dataset.py training/training_data.csv --test HERC2A,INPP5DA,NSP14,IMPA1A -s

```
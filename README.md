# panddaproject

# dependencies
```shell
conda create -n pandda_env python=3.9.13 biopython=1.78
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
conda install -c conda-forge -y gemmi=0.4.5
conda install -c conda-forge -y pandas=1.4.3 plotly-5.10.0 matplotlib=2.2.3 fire=0.4.0
pip install notebook
pip install -e .
```

#splitting dataset into training and test sets
```shell
python mldataprep/split_dataset.py training/training_data.csv --test HERC2A,INPP5DA,NSP14,IMPA1A -s

```
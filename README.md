
# PCAM MLOps Project - Groep [NUMMER]



## Installation

1. Clone deze repository.

2. Maak een virtual environment: `python -m venv venv`.

3. Activeer de environment: `source venv/bin/activate`.

4. Installeer dependencies: `pip install -r requirements.txt`.



## Data Setup

Plaats de PCAM HDF5-bestanden in een map genaamd `data/`.

De benodigde bestanden (zoals te vinden op Snellius) zijn:

- `camelyonpatch_level_2_split_train_x.h5`

- `camelyonpatch_level_2_split_train_y.h5`



## Training

Om ons beste model te reproduceren, gebruik het volgende commando:

`python src/train.py --config configs/best_model.yaml`

*Verwachte prestatie:* Validatie Accuracy van ca. XX%.



## Inference

Om een enkele voorspelling te doen met het opgeslagen model:

`python inference.py`


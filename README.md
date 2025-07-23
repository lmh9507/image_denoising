# Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images


## Python Requirements

This code was tested on:

- Python 3.7
- Pytorch 1.3

## Training

To train a network, run:

```bash
bash train.sh
```

## Data directory tree
```
data
├── clean
│   ├── sem0000_00.tif
│   ├── sem0000_01.tif
│   ├── sem0000_02.tif
│   └── ⋯
└── noise
    ├── sem0000_00.tif
    ├── sem0000_01.tif
    ├── sem0000_02.tif
    └── ⋯
```

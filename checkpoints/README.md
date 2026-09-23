# Pretrained Model Checkpoints

This directory is organized to store pretrained model checkpoints for the three evaluation benchmarks:

```
checkpoints/
├── abdomen_ctct/
│   ├── ecdp_best.pt          # Proposed ECDP model (Abdomen CT-CT, DSC: 52.56%)
│   └── eddp_baseline.pt      # EDDP baseline model (Abdomen CT-CT, DSC: 51.86%)
├── ixi/
│   ├── ecdp_best.pt          # Proposed ECDP model (IXI Brain MRI, DSC: 75.34%)
│   └── eddp_baseline.pt      # EDDP baseline model (IXI Brain MRI, DSC: 74.80%)
└── lpba40/
    ├── ecdp_best.pt          # Proposed ECDP model (LPBA40 Brain MRI, DSC: 73.84%)
    └── eddp_baseline.pt      # EDDP baseline model (LPBA40 Brain MRI, DSC: 73.20%)
```

## Download Links

Download the pretrained weights from Google Drive or GitHub Releases and place the `.pt` files into their corresponding directories above:
- **Google Drive**: `[Google Drive Link - Coming Soon]`
- **GitHub Releases**: `https://github.com/goghfan/ECDP/releases`

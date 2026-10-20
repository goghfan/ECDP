# Pretrained Model Checkpoints

This directory is organized to store pretrained model checkpoints for the three evaluation benchmarks:

```
checkpoints/
├── abdomen_ctct/
│   ├── ecdp_best.pt          # Proposed ECDP model (AbdomenCTCT_53.15.pt, DSC: 53.15% / 0.532)
│   └── eddp_baseline.pt      # EDDP baseline model (Abdomen CT-CT, DSC: 51.59% / 0.516)
├── ixi/
│   ├── ecdp_best.pt          # Proposed ECDP model (IXI_76.00.pt, DSC: 76.00% / 0.760)
│   └── eddp_baseline.pt      # EDDP baseline model (IXI Brain MRI, DSC: 75.92% / 0.759)
└── lpba40/
    ├── ecdp_best.pt          # Proposed ECDP model (LPBA40_74.02.pt, DSC: 74.02% / 0.740)
    └── eddp_baseline.pt      # EDDP baseline model (LPBA40 Brain MRI, DSC: 73.98% / 0.740)
```

## Download Links

Download the pretrained weights from Google Drive and place the `.pt` files into their corresponding directories above (you can keep the downloaded filename like `AbdomenCTCT_53.15.pt` or rename to `ecdp_best.pt`):
- **Google Drive**: [ECDP Pretrained Checkpoints](https://drive.google.com/drive/folders/1kuBD0TDpgpmdYJEnXTW1_Pvh0HdhGJhw?usp=sharing)

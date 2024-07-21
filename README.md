# RUE-Net: Advancing Underwater Vision with Live Image Enhancement
Implementation of the paper *[RUE-Net: Advancing Underwater Vision with Live Image Enhancement](https://ieeexplore.ieee.org/abstract/document/10537222)*

## The model structure diagram of RUE-Net
![RUE_Net2 drawio](https://github.com/GuocunWang/RUE-Net/assets/103011611/11973434-ef72-449b-a09a-dc71b4b7cc6c)

## Ablation study of RUE-Net

![消融实验](https://github.com/GuocunWang/RUE-Net/assets/103011611/ca6ef2c3-c13a-4737-98d3-052afa515bf1)


## Recommended environment:
```
Python 3.8
torch 1.8.0
pytorch-ssim
```

## Train the Model
```
python training.py
```

## Test the Model
```
python test.py
```

## Citation

If you find our work useful, please consider citing the paper.

```
@article{wang2024rue,
  title={RUE-Net: Advancing Underwater Vision with Live Image Enhancement},
  author={Wang, Guocun and Chen, Chen and Xu, Hongli and Ru, Jingyu and Wang, Shuai and Wang, Zhenglong and Liu, Zhaofeng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
## Acknowledgement

We are very grateful for the excellent work [Shallow-UWnet](https://github.com/mkartik/Shallow-UWnet), which has provided the basis for our framework.
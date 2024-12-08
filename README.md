## BiEnNet: Bilateral enhancement network with signal-to-noise ratio fusion for lightweight generalizable low-light image enhancement (Sci. Rep 2024) [(paper)](https://rdcu.be/d2gRo)

### BiEnNet's network structure
<div align=center><img src="https://github.com/2665207323/BiEnNet/blob/main/BiEnNet/img/BiEnNet.jpg"/></div>

### Requirements
- python >=3.8
- numpy
- torch
- torchvision

### Datesets
We provide download links for training and testing datasets as well as datasets used by NIQE indicators[(Google Drive)](https://drive.google.com/file/d/1jP84rkPEwRmSDgzG9OWsVwRvsEA5cWld/view?usp=drive_link)[(baiduyun)](https://pan.baidu.com/s/1QqQSFwcAOiofp0XcVJ9QQw?pwd=eman)


### Training useing the training datasets

Check the parameter settings, model and image pathes in train.py, and then run:
```
train.py
```

### Testing for testing datasets

Check the model and image pathes in test.py, and then run:
```
test.py
```

### Training the NIQE model

To train the NIQE model, you need to prepare the training dataset of NIQE.

Check the dataset path in NIQE, and then run this file use matlab:
```
train_NIQE.m
```

### Testing the NIQE value of testing datasets

Check the model and image pathes in test_mix.py and NIQE.m, and then run:
```

NIQE.m
```

### Testing the CIEDE2000 value of testing datasets

Check the model and image pathes in CIEDE2000.py, and then run:
```
CIEDE2000.py
```

## Citation

If you find PairLIE is useful in your research, please cite our paper:

```
@article{wangBilateralEnhancementNetwork2024b,
  title = {Bilateral Enhancement Network with Signal-to-Noise Ratio Fusion for Lightweight Generalizable Low-Light Image Enhancement},
  author = {Wang, Junfeng and Huang, Shenghui and Huo, Zhanqiang and Zhao, Shan and Qiao, Yingxu},
  year = {2024},
  month = nov,
  journal = {Scientific Reports},
  pages = {29832},
  issn = {2045-2322},
  doi = {10.1038/s41598-024-81706-2},
}
```

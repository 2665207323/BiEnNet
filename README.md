## BiEnNet: Bilateral enhancement network with signal-to-noise ratio fusion for lightweight generalizable low-light image enhancement (Sci. Rep 2024) [(paper)](https://rdcu.be/d2gRo)

### BiEnNet's network structure
<div align=center><img src="https://github.com/2665207323/BiEnNet/blob/main/BiEnNet/img/BiEnNet.jpg" height = "60%" width = "60%"/></div>

### Requirements
- numpy
- torch
- torchvision

### Datesets
We provide download links for training and testing datasets as well as datasets used by NIQE indicators[(Google Drive)](https://rdcu.be/d2gRo)[(aliyun)](https://rdcu.be/d2gRo)


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
test_mix.py
NIQE.m
```

### Testing the CIEDE2000 value of testing datasets

Check the model and image pathes in CIEDE2000.py, and then run:
```
CIEDE2000.py
```

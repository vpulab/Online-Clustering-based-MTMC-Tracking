## Online-MTMC-vehicle-tracking

Paper:  https://link.springer.com/article/10.1007/s11042-022-11923-2



# Setup & Running
**Requirements**

The repository has been tested in the following software.
* Ubuntu 16.04
* Python 3.7
* Anaconda
* Pycharm

**Anaconda environment**

To create and setup the Anaconda Envirmorent run the following terminal command from the repository folder:
```
$ conda env create -f env_MTMC.yaml
$ conda activate env_MTMC
```

**Clone repository**

```
$ git clone https://github.com/elun15/Online-MTMC-vehicle-tracking.git
```

**Download AIC19 dataset**

The dataset can be downloaded at https://www.aicitychallenge.org/track1-download/

**Prepare AIC19 dataset**

Move the downloaded folders *aic19-track1-mtmc/train* and *aic19-track1-mtmc/test* to the *./datasets/AIC19/* repository folder.

Preprocess the data to extract the images from the .avi files by running:

```
python preprocess_data.py
```


The set of data can be changed, by default it will preprocess */test/S02* scenario.


**Download pretrained model**

The model weights trained on AIC19 S01 scenario can be downloaded at:
http://www-vpu.eps.uam.es/publications/Online-MTMC-Tracking/ResNet50_AIC20_VERI_layer5_imaugclassifier_latest.pth.tar


Place the weights file under *./models/*

Training details can be found in the paper.


**Running**

To run the tracking algorithm over the S02 scenario run:

```
python main.py --ConfigPath ./config/config.yaml  
```



# Citation

If you find this code and work useful, please consider citing:
```
@article{luna2022online,
  title={Online clustering-based multi-camera vehicle tracking in scenarios with overlapping FOVs},  
  author={Luna, Elena and SanMiguel, Juan C and Mart{\'\i}nez, Jos{\'e} M and Escudero-Vi{\~n}olo, Marcos},  
  journal={Multimedia Tools and Applications},
  pages={1--21},  
  year={2022},  
  publisher={Springer}
}
```




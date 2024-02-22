# Iris-LAHNet: A Lightweight Attention-guided High-resolution Network for Iris Segmentation and Localization
## Introduction
This is the code for IRIS-LAHNET. Iris-LAHNet is composed of a stem, a basic backbone, Pyramid Dilated Convolution (PDC) blocks, Cascaded Attention-guided Feature Fusion Module (C-AGFM), and auxiliary heads. The basic backbone is a tiny high-resolution network. The introduction of PDC blocks and C-AGFM helps to extract multi-scale features from multi-resolution images and reduce noise. In addition, we introduce three auxiliary heads with edge heatmaps, which output auxiliary loss to help model training and enhance attention to single pixels of the edge. It helps to compensate for the neglect of localization tasks during multi-task training. Experiments on four datasets show that our model achieves the lightest while ensuring segmentation and localization results.

![pic1](https://github.com/YAN-0802/LAHNet/blob/main/pics/LAHNet.jpg)

## Datasets and Main Results
 We compare our model with other state-of-the-art methods on four public datasets. Refer to [IrisParseNet](https://doi.org/10.1109/TIFS.2020.2980791) for related settings.

### Result on CASIA-Iris-Distance
<table>
	<tr>
	    <th rowspan="2">Method</th>
	    <th colspan="3">Segmentation</th>
	    <th colspan="3">Localization</th> 
      <th rowspan="2">Params(M)</th>
      <th rowspan="2">FLOPs(G)</th>
	</tr >
  	<tr>
	    <th>E1(%)</th>
	    <th>mIoU(%)</th> 
      <th>F1(%)</th>
      <th>Inner(%)</th>
      <th>Outer(%)</th>
      <th>mHdis(%)</th>
	</tr >
	<tr>
	    <th><a href="https://ieeexplore.ieee.org/document/7410793">RTV-L1</a></th>
	    <td>0.68</td>
      <td>78.25</td>
      <td>87.55</td>
      <td>0.7046</td>
      <td>1.2457</td>
      <td>0.9751</td>
      <td>\</td>
      <td>\</td>
	</tr>
 	<tr>
	    <th><a href="https://doi.org/10.1109/ICB.2016.7550055">MFCNs</a></th>
	    <td>0.59</td>
      <td>\</td>
      <td>93.09</td>
      <td>\</td>
      <td>\</td>
      <td>\</td>
      <td>21.68</td>
      <td>156.35</td>
	</tr>
	 	<tr>
	    <th><a href="https://doi.org/10.1007/978-3-319-24574-4 28">U-Net</a></th>
	    <td>0.56</td>
      <td>\</td>
      <td>\</td>
      <td>0.6129</td>
      <td>1.1478</td>
      <td>0.8804</td>
      <td>31.06</td>
      <td>225.94</td>
	</tr>
 	 	<tr>
	    <th><a href="https://doi.org/10.1016/j.patrec.2018.12.021">CNNHT</a></th>
	    <td>0.56</td>
      <td>86.58</td>
      <td>92.27</td>
      <td>1.1973</td>
      <td>2.0251</td>
      <td>1.6112</td>
      <td>61.87</td>
      <td>144.79</td>
	</tr>
 	 	<tr>
	    <th><a href="https://doi.org/10.1109/TIFS.2020.2980791">IrisParseNet</a></th>
	    <td>0.41</td>
      <td>89.53</td>
      <td>94.25</td>
      <td>0.6874</td>
      <td>0.8662</td>
      <td>0.7768</td>
      <td>31.68</td>
      <td>263.56</td>
	</tr>
 	 	<tr>
	    <th><a href="https://doi.org/10.1109/IJCB54206.2022.10007944">HTU-Net</a></th>
	    <td>0.43</td>
      <td>\</td>
      <td>\</td>
      <td>0.5381</td>
      <td>0.9702</td>
      <td>0.7541</td>
      <td>22.27</td>
      <td>239.74</td>
	</tr>
 	 	<tr>
	    <th><a href="">Iris-LAHNet</a></th>
	    <td>0.36</td>
      <td>90.78</td>
      <td>95.15</td>
      <td>0.4915</td>
      <td>0.8990</td>
      <td>0.6953</td>
      <td>0.27</td>
      <td>5.57</td>
	</tr>
</table>

  ### Result on MICHE-I
  ![pic2](https://github.com/YAN-0802/LAHNet/blob/main/pics/miche.jpg)

  ### Result on UBIRIS.v2
  ![pic3](https://github.com/YAN-0802/LAHNet/blob/main/pics/ub.jpg)

  ### Result on CASIA-Iris-Mobile-V1.0
  ![pic4](https://github.com/YAN-0802/LAHNet/blob/main/pics/m1.jpg)

## Requirements
+ Python = 3.7.13
+ Pytorch = 1.13.0+cu117
+ Numpy = 1.21.6
+ pandas = 1.3.5
+ mmcv = 1.7.0
+ torchvision = 0.14.0+cu117
Note the version number is for reference only, you can choose the version number that is suitable for your system.

## Quick Start
### Prepare the data
Thanks for the selfless contribution of previous work， we adopt the dataset provided in [IrisParseNet](https://doi.org/10.1109/TIFS.2020.2980791) for related settings.

### Prepare the models
Iirs-LAHNet trains the model on different datasets. All the model weights are saved in the `logging.pth` file under the [`./LAHNet/experiments`](https://github.com/YAN-0802/LAHNet/tree/main/LAHNet/experiments) folder. 

 Let's take training or testing on MICHE as an example. 
### Training
you can run training as following:

```
CUDA_VISIBLE_DEVICES=gpuid python train.py
```
Note: You need to change the associated path in `train.py` to your own path. The trained model is saved in the appropriate `./~/checkpoints/` folder in the `./LAHNet/experiments/` folder. Of course, you can also change the location and name of the save.

### Test
you can run testing as following:

```
CUDA_VISIBLE_DEVICES=gpuid python test.py
```
Note: You need to change the associated path in `test.py` to your own path. The results will be saved in the appropriate `./~/checkpoints/` folder in the `./LAHNet/experiments/` folder. Of course, you can also change the location and name of the save.

### Post-processing
Finally, the output can be visualized：

```
CUDA_VISIBLE_DEVICES=gpuid python postprocess.py
```
This step can be ignored because important results have been saved in the `checkpoints` folder after running `test.py`.

If you want to experiment with other datasets, just change the name of the dataset in the `.py` file.

## Acknowledgement
The research is supported by “ the Fundamental Research Funds for the Central Universities ” under Grant Agreement No. N2105009, National Natural Science Foundations of China under Grant Agreement No.61703088.

## Citation
If you use our code or models in your research, please cite with:
```
```

If you have any questions, please contact us by email.

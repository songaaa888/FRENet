# Deepfake Detection via Feature Refinement and Enhancement Network for Consumer Electronics
<b> Authors: Weicheng Song</a>, Siyou Guo</a>, Dan Zhang*</a>,  Xianxun Zhu*</a>, Mingliang Gao</a>  </b>


> News:
> OUR LATEST WORK:  We have proposed a network model aimed at addressing the issue of high-risk transactions involving deepfake-generated faces in facial payment systems.  We will soon release all the codes implemented by the DeepfakeBench codebase.


### Training 

<a href="#top">[Back to top]</a>

To run the training code, you should first download the pretrained weights for the corresponding **backbones** (These pre-trained weights are from ImageNet). You can download them from [Link](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip). After downloading, you need to put all the weights files into the folder `./training/pretrained`.

Then, you should go to the `./training/config/detector/` folder and then Choose the detector to be trained. For instance, you can adjust the parameters in [`xception.yaml`](./training/config/detector/xception.yaml) to specify the parameters, *e.g.,* training and testing datasets, epoch, frame_num, *etc*.

After setting the parameters, you can run with the following to train the Xception detector:

```
python training/train.py --detector_path 
```

You can also adjust the training and testing datasets using the command line, for example:

```
python training/train.py --detector_path   --train_dataset "FaceForensics++" --test_dataset "Celeb-DF-v1"
```

By default, the checkpoints and features will be saved during the training process. If you do not want to save them, run with the following, for example:

```
python training/train.py --detector_path ./training/config/detector/xception.yaml --train_dataset "FaceForensics++" --test_dataset "Celeb-DF-v1" --no-save_ckpt --no-save_feat
```

### Test

If you want to produce the results, you can use the the [`test.py`](./training/test.py) code for evaluation. Here is an example:

```
python3 training/test.py --detector_path /home/weicheng/DeepfakeBench/training/config/detector/spsl.yaml --test_dataset "Celeb-DF-v1" --weights_path /home/weicheng/Experiments/Deep/DeepfakeBench/logs/spsl_2025-04-06-14-36-171/test/Celeb-DF-v1/ckpt_best.pth


### Datasets split ratios

For intra-dataset validation, FF++  dataset serves as the training dataset, while FF++ dataset and its four subsets are used as testing datasets. The explicit ratios are shown in：

| dataset       | Training samples | Testing samples | Training ratio | Testing ratio |
|:-------------|:---------------:|:---------------:|:--------------:|--------------:|
| FF++         | 3596            | 700             | 83.71%         | 16.29%        |
| DF           | 1438            | 280             | 83.70%         | 16.30%        |
| F2F          | 1438            | 280             | 83.70%         | 16.30%        |
| FS           | 1439            | 280             | 83.71%         | 16.29%        |
| NT           | 1438            | 280             | 83.70%         | 16.30%        |


For cross-dataset validation, FF++ dataset serves as the training dataset, with Celeb-DF-v1, Celeb-DF-v2, DFDCP , and FaceShifter  datasets as testing datasets. The slash expresses that it was not used in the cross-dataset validation. The explicit ratios are shown in：
| dataset       | Training samples | Testing samples | Training ratio | Testing ratio |
|:-------------|:---------------:|:---------------:|:--------------:|--------------:|
| FF++         | 3596            | 700             | 83.71%         | ——            |
| Celeb-DF-v1  | 33308           | 3136            | ——             | 8.60%         |
| Celeb-DF-v2  | 189301          | 16420           | ——             | 7.98%         |
| DFDCP        | 822             | 230             | ——             | 21.86%        |
| FaceShifter  | 1438            | 280             | ——             | 16.30%        |
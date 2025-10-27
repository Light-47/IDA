# IDA
> Implementation for paper **Intermediate Domain-guided Adaptation for Unsupervised Chorioallantoic Membrane Vessel Segmentation**

<!-- Project Views -->
<p align="left">
  <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLight-47%2FIDA&count_bg=%2379C83D&title_bg=%23555555&title=views&edge_flat=false" alt="IDA Views" />
</p>

## Requirements
* matplotlib==3.5.1
* numpy==1.23.4
* opencv-python==4.10.0.84
* scipy==1.5.4
* seaborn==0.13.2
* torch==1.13.1+cu116
* torchaudio==0.13.1+cu116
* torchvision==0.14.1+cu116

## Training
If you want to train your own model, please run: `python train.py`

You can modify training parameters in `scripts/configs/DRIVE2CAM.yml`, `domain_adaptation/config_vessel.py` and `parser_train.py`

The training weights will be saved in `scripts/experiments/`
## Testing
If you have ownd the model weights, run `generate_results_new.py` to get the predictions.

And the evaluation metrics(AUC, ACC, SE, SP, DICE) can be computed by running `analyze_results_test.py`.

## Dataset and Weights
For the CAM_DB dataset and model weights, please download from this [link](https://drive.google.com/drive/folders/1ixgoOKNaco7yQKrc0doH444L8pJYoqar?usp=sharing) (2025.3)

Now, an expanded CAM_DB dataset is available. We will release it to the public once the paper is accepted. (2025.7)
## Acknowledgement
We would like to thank [MPSCL](https://github.com/TFboys-lzz/MPSCL), [VesselSeg](https://github.com/lee-zq/VesselSeg-Pytorch), [WNet](https://github.com/agaldran/lwnet) for their valuable models and ideas.
## Citation
```
@article{song2025intermediate,
  title = {Intermediate Domain-guided Adaptation for Unsupervised Chorioallantoic Membrane Vessel Segmentation},
  author = {Song, Peng and Wang, Zhi and Yao, Peng and others},
  journal = {arXiv preprint arXiv:2503.03546},
  year = {2025}
}
```

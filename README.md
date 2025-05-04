# IDA
> Implementation for paper **Intermediate Domain-guided Adaptation for Unsupervised Chorioallantoic Membrane Vessel Segmentation**
<img src="https://badges.toozhao.com/badges/01JSZY7BEB0WS6GKTKA47X4AEC/green.svg" />
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
For the CAM_DB dataset and model weights, please download from this [link](https://drive.google.com/drive/folders/1ixgoOKNaco7yQKrc0doH444L8pJYoqar?usp=sharing) 
## References
* [MPSCL](https://github.com/TFboys-lzz/MPSCL)
* [VesselSeg](https://github.com/lee-zq/VesselSeg-Pytorch)
* [WNet](https://github.com/agaldran/lwnet)

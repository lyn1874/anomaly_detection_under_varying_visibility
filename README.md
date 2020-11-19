#### Anomaly detection under varying illuminations and weather conditions

The goal of this repository is to evaluate the performance of the state-of-the-art anomaly detectors on the scenes with varying visibility. We consider the visibility changes are caused by illumination changes (day-night) and weather switches (sunny-rainy). The state-of-the-art anomaly detectors that we use are listed below:

- [Future Frame Prediction for Anomaly Detection -- A New Baseline (FFP-MC)](https://arxiv.org/abs/1712.09867)
- [Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf)
- [Learning Memory-guided Normality for Anomaly Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.pdf)

##### Prepare the datasets

Most of the anomaly detection benchmark datasets contain relatively clean frames, which are usually collected under a well-controlled environment (sunny and similar times of the data). These frames are not very representative of the surveillance cameras in a city-context where the illuminations and weathers are frequently changing. Thus, to quantitative evaluate the effectiveness of the anomaly detectors under scenes with varying visibility, we augment the standard Avenue dataset by altering illuminations and adding raindrops.

- Download the dataset

  `./prepare_dataset.sh`

- Augment the dataset

  `python3 aug_data.py --rain_type heavy --bright 4 --datapath datadir`

##### Train and evaluate the models

In each subfolder, first run `./requirement.sh` to install the required packages. Then run `./run.sh` to train and model and `./eval.sh` to evaluate the model. Note the `datapath` argument in each subfolder needs to be manually defined



#### References

https://github.com/feiyuhuahuo/Anomaly_Prediction

https://github.com/cvlab-yonsei/MNAD

https://github.com/donggong1/memae-anomaly-detection



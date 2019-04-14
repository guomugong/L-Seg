# L-Seg
L-Seg: An End-to-End Unified Framework for Multi-lesion Segmentation of Fundus Images

Please read our [paper] (https://doi.org/10.1016/j.neucom.2019.04.019) for more details!
### Introduction:
Diabetic retinopathy and diabetic macular edema are the two leading causes for blindness in working-age people, and the quantitative and qualitative diagnosis of these two diseases usually depends on the presence and areas of lesions in fundus images. The main related lesions include soft exudates, hard exudates, microaneurysms, and haemorrhages. However, segmentation of these four kinds of lesions is difficult due to their uncertainty in size, contrast, and high interclass similarity. Therefore, we aim to design a multi-lesion segmentation model.
We have designed the first small object segmentation network (L-Seg) that can segment the four kinds of lesions simultaneously. Taking into account that small lesion regions could not response at high level of network, we propose a multi-scale feature fusion method to handle this problem. In addition, when considering the cases of both class-imbalance and loss-imbalance problems, we propose a multi-channel bin loss.
We have evaluated L-Seg on three fundus datasets including two publicly available datasets - IDRiD and e-ophtha and one private dataset - DDR. Extensive experiments have demonstrated that L-Seg achieves better performance in small lesion segmentation than other deep learning models and traditional methods. Specially, the mAUC score of L-Seg is over 16.8%, 1.51% and 3.11% higher than that of DeepLab v3+ on IDRiD, e-ophtha and DDR datasets, respectively. Moreover, our framework shows competitive performance compared with top-3 teams in IDRiD challenge.

## Usage:
```
layer {
  name: "loss"
  type: "MultiChannelBinSigmoidCrossEntropyLoss"
  bottom: "pred"
  bottom: "label"
  top: "loss"
  loss_weight: 1.0
  mcbsce_loss_param {
    key: 10
    num_label: 4
  }
}

```
## License
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![Badge](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/zh_CN)

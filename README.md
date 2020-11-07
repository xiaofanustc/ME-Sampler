# Motion-Excited Sampler: Video Adversarial Attack with Sparked Prior
This repository contains code for the paper:
#### Motion-Excited Sampler: Video Adversarial Attack with Sparked Prior

Hu Zhang, Linchao Zhu, Yi Zhu, Yi Yang

[[Arxiv](https://arxiv.org/abs/2003.07637)]
[[Slides](https://drive.google.com/file/d/1J4pN5nM2jfmRuONhX_GAS0Ts_0IX5c_z/view?usp=sharing)]
[[Demo Video](https://drive.google.com/file/d/1v0Zfruy_gEQZlG35hdqPw3BwPdaUhN9F/view)]

ReLER, University of Technology Sydney, NSW; Amazon Web Services

#### Abstract
Deep neural networks are known to be susceptible to ad- versarial noise, which is tiny and imperceptible perturbations. 
Most of previous work on adversarial attack mainly focus on image models, while the vulnerability of video models is less explored. 
In this paper, we aim to attack video models by utilizing intrinsic movement pattern and regional relative motion among video frames. 
We propose an effective motion- excited sampler to obtain motion-aware noise prior, which we term as sparked prior. 
Our sparked prior underlines frame correlations and uti- lizes video dynamics via relative motion. 
By using the sparked prior in gradient estimation, we can successfully attack a variety of video clas- sification models with fewer number of queries. 
Extensive experimental results on four benchmark datasets validate the efficacy of our proposed method.

### Requirements:
- python 3.6
- [pytorch 1.0.1](https://pytorch.org/)
- [coviar](https://github.com/chaoyuaw/pytorch-coviar)
- [mxnet 1.5.0](https://mxnet.apache.org/versions/1.6/) 
- [gluoncv 0.6.0](https://gluon-cv.mxnet.io/contents.html)
### Dataset
Something-Something v2: video is split into frames by `video2frames.py` and change the path in `run_smth_i3d.sh`.
### Attacked models
We use existing I3D and TSN2D models from [gluoncv](https://gluon-cv.mxnet.io/model_zoo/action_recognition.html), download [[here](https://drive.google.com/drive/folders/10lWG0kEUjbsEeJOWVo-WhExyu5JS0Jid?usp=sharing)]. You can replace this part with other models.

### Mpeg video generation
When use [coviar](https://github.com/chaoyuaw/pytorch-coviar) to extract motion vector, first convert original video to mpeg format: 

run `bash reencode_smth_smth.sh`.

### Attacking
run `bash run_smth_i3d.sh` or `bash run_smth_tsn.sh`

**Reminder: when attacking, we impose noise after normalize pixels to 0-1 but before mean and std normalization, thus we need to split previous operations of transformation.**
### License
This project is licensed under the license found in the [LICENSE](https://github.com/xiaofanustc/ME-Sampler/blob/master/LICENSE) file in the root directory.

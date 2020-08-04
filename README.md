# ME-Sampler
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
python 3
[pytorch 1.0.1](https://pytorch.org/)
[coviar](https://github.com/chaoyuaw/pytorch-coviar)
### Dataset
Something-Something v2: video is split into frames by xx and change the path in `run_smth_i3d.sh`.
### Mpeg video generation
When use [coviar](https://github.com/chaoyuaw/pytorch-coviar) to extract motion vector, change original video to mpeg format: run `bash reencode_smth_smth.sh`.

### Attacking
run bash `run_smth_i3d.sh` or `bash run_smth_tsn.sh`

### License
This project is licensed under the [LICENSE](https://github.com/xiaofanustc/ME-Sampler/blob/master/LICENSE) found in the LICENSE file in the root directory.

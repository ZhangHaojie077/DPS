# DPS #
Source code for NeurIPS 2022 Long paper: [Fine-Tuning Pre-Trained Language Models Effectively by Optimizing Subnetworks Adaptively](https://arxiv.org/pdf/2211.01642.pdf)

## 1. Environments ##
+ python: 3.8.10
+ CUDA Version: 11.4 

## 2. Dependencies ##
+ torch==1.8.0
+ datasets==1.8.0
+ transformers==4.7.0

## 3. Training ##
```
cd script
```
### 3.1 Fine-tuning on GLUE ###
```
bash run_glue.sh
```
### 3.2 Fine-tuning on out domain ###
```
bash run_out_of_domain.sh
```
### 3.3 Fine-tuning on low resource ###
```
bash run_low_resource.sh
```

## 4. Citation ##
If you use this work or code, please kindly cite the following paper:
```
@inproceedings{haojiefine,
  title={Fine-Tuning Pre-Trained Language Models Effectively by Optimizing Subnetworks Adaptively},
  author={Haojie, Zhang and Li, Ge and Li, Jia and Zhang, Zhongjin and Zhu, Yuqi and Jin, Zhi},
  booktitle={Advances in Neural Information Processing Systems}
}
```
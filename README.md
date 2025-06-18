## 《Compress Any Segment Anything Model (SAM)》

![Static Badge](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg) ![Static Badge](https://img.shields.io/badge/License-Apache--2.0-orange.svg) ![GitHub Repo stars](https://img.shields.io/github/stars/Juntongkuki/Birkhoff-Model-Compression?style=flat&logo=github) 

[📄[paper](-)] [📍[Github](https://github.com/Juntongkuki/Birkhoff-Model-Compression)]


## News 📢 
* **2025/06/18** : Big day! 🥰 Our **Birkhoff** code has officially gone open source. ✨ We'll be pushing updates as the saga continues. Thanks for watching, supporting! 😄 Moreover, welcome to read our previous work [**《Hyper-Compression: Model Compression via Hyperfunction》**](https://github.com/Juntongkuki/Hyper-Compression.git), which is the theoretical foundation of **Birkhoff**.



## Introduction
Due to the excellent performance in yielding high-quality, zero-shot segmentation, Segment Anything Model (SAM) and its variants have been widely applied in diverse scenarios such as healthcare and intelligent manufacturing. Therefore, effectively compressing SAM and its variants has become an increasingly pressing practical need. Unlike quantization, pruning, distillation, and low-rank decomposition, we propose *Birkhoff* algorithm for systematically compressing SAM and its variants. Specifically, *Birkhoff* introduces a novel compression algorithm: Hyper-Compression, whose core principle is to find a dense trajectory to turn a high-dimensional parameter vector into a low-dimensional scalar. Furthermore, *Birkhoff* designs a dedicated linear layer operator, HyperLinear, to fuse decompression and matrix multiplication to significantly accelerate inference of the compressed SAMs. *Birkhoff* is a universal, data-free, fast, and high-accuracy-compression-ratio compression algorithm. Extensive experiments on 18 SAMs in the COCO, LVIS, and SA-1B datasets show that *Birkhoff* performs consistently and competitively in compression time, compression ratio, post-compression performance, and inference speed. For example, *Birkhoff* can achieve a compression ratio of 5.17× on SAM2-B, with less than 1% performance drop without using any fine-tuning data. Moreover, the compression is finished within 60 seconds for all models.


<div align="center">
  <img src="assets/overview.png" width="95%" />
</div>
<p align="center"><b>Fig. </b> Overview of our Birkhoff compression framework.</p>


### The proposed *Birkhoff* enjoys the following merits: 💡

* **V**ersatility across model types 

* **A**gility in model deployment

* **F**aithfulness to the original model

* **C**ompactness in model size


## 🥳 *Birkhoff* 🍾

### Experimental Results
<div align="center">
  <img src="assets/seg_every.png" width="85%" />
</div>
<p align="center"><b>Fig. </b> We select SAM-B, SAM-L, and SAM-H for a visual comparison of the Segment Everything task before and after Birkhoff compression.</p>


## Requirements
The code requires `python>=3.8.0` and we use `torch==2.4.0` and `torchvision==0.19.0`. For the usage of **HyperLinear** operator, `triton==3.0.0` is also required.  
- python == 3.8.0
- torch==2.4.0
- torchvision==0.19.0
- albumentations==2.0.8
- numpy==1.24.3
- safetensors==0.5.3
- triton==3.0.0

## Usage

1. Download the [checkpoint](https://drive.google.com/file/d/192zoxFf5MUELdUzFCgcv4l52wFKmq9pa/view?usp=drive_link) of SAM-HQ-H into the directory of *./sam_family/checkpoints/*.

2. Run the demo code for compressing SAM-HQ-H.

```
python main.py --mode='encode' --model_name='sam_hq_vit_h' --checkpoint_path='/home/ET/jtfan/MyData/checkpoints/sam_hq_vit_h.pth'
```

3. Run the demo code for segment everything task, the results will be saved into *./test_data/seg_results*.
```
python main.py --mode='inference' --model_name='sam_hq_vit_h'
```



## Acknowledgements
We thank the following projects: [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/sam2.git), [SAM-HQ](https://github.com/SysCV/sam-hq.git), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [MobileSAMv2](https://github.com/ChaoningZhang/MobileSAM.git), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM.git), [EfficientSAM](https://github.com/yformer/EfficientSAM.git), [TinySAM](https://github.com/xinghaochen/TinySAM.git), [MedSAM](https://github.com/bowang-lab/MedSAM.git).

[//]: # (## Citation)

[//]: # (```bibtex)

[//]: # (@article{})

[//]: # (```)

## License

This project is licensed under <a rel="license" href="License.txt"> Apache License 2.0</a>. Redistribution and use should follow this license.
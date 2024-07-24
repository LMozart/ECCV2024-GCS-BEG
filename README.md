# Connecting Consistency Distillation to Score Distillation for Text-to-3D Generation
[Zongrui Li*](https://github.com/LMozart), [Minghui Hu*](https://mhh0318.github.io/), [Qian Zheng✉️](https://person.zju.edu.cn/zq), [Xudong Jiang](https://personal.ntu.edu.sg/exdjiang/)

*: Equal Contribution. ✉️: Corresponding author.
<div align=center>
<img src="assets/teaser.gif" width="90%"/> 
</div>

**[[Project Page]](https://zongrui.page/ECCV2024-GCS-BEG/)** **[[Arxiv]](https://arxiv.org/abs/2407.13584)**

## 🚩 Brief
We present an improved 2D-to-3D distillation method using theories from consistency distillation.

## 💻 Installation

```shell
conda create -n gcs python=3.9.16 cudatoolkit=11.8
conda activate gcs
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
pip install submodules/point-e/
pip install tensorboard
```

## 🥊 Training
```shell
python train.py --opt configs/full_model/cat_armor.yaml
# train script
sh scripts/train_0.sh
```

## 🧾 Todo List
- [x] Release the basic training codes

## 📖 Citation 
```
@article{li2024gcs,
  title={Connecting Consistency Distillation to Score Distillation for Text-to-3D Generation},
  author={Li, Zongrui and Hu, Minghui and Zheng, Qian and Jiang, Xudong},
  journal={arXiv preprint arXiv:2407.13584},
  year={2024}
}
```

## 🙏 Acknowledgement
Our work is developed on [LucidDreamer](https://github.com/EnVision-Research/LucidDreamer/). Thanks for their contribution to this task!
<div align="center">

# InternDataEngine: A simulation-based data generation engine designed for robotic learning. 

</div>

[![Paper InternData-A1](https://img.shields.io/badge/Paper-InternData--A1-red.svg)](https://arxiv.org/abs/2511.16651)
[![Paper Nimbus](https://img.shields.io/badge/Paper-Nimbus-red.svg)](https://arxiv.org/abs/2601.21449)
[![Paper InternVLA-M1](https://img.shields.io/badge/Paper-InternVLA--M1-red.svg)](https://arxiv.org/abs/2510.13778)
[![Data InternData-A1](https://img.shields.io/badge/Data-InternData--A1-blue?logo=huggingface)](https://huggingface.co/datasets/InternRobotics/InternData-A1)
[![Data InternData-M1](https://img.shields.io/badge/Data-InternData--M1-blue?logo=huggingface)](https://huggingface.co/datasets/InternRobotics/InternData-M1)
[![Docs](https://img.shields.io/badge/Docs-TBD-lightgrey.svg)](#)

## 💻 About

InternDataEngine is a data-centric engine for embodied AI that powers large-scale model training and iteration.  
Built on NVIDIA Isaac Sim, it unifies high-fidelity physical interaction from InternData-A1, semantic task and scene generation from InternData-M1, and high-throughput scheduling from the Nimbus framework to deliver realistic, task-aligned, and massively scalable robotic manipulation data.

- **More realistic physical interaction**: Unified simulation of rigid, articulated, deformable, and fluid objects across single-arm, dual-arm, and humanoid robots, enabling long-horizon, skill-composed manipulation that better supports sim-to-real transfer.
- **More task-aligned data generation**: LLM-driven task and instruction generation with task-oriented scene graphs (ToSG), producing structured scenes and rich multi-modal annotations (boxes, keypoints, trajectories) for complex instruction-following and spatial reasoning.
- **More efficient large-scale production**: Nimbus-powered asynchronous pipelines that decouple planning, rendering, and storage, achieving 2–3× end-to-end throughput, cluster-level load balancing and fault tolerance for billion-scale data generation.

## 🔥 Latest News 

- **[2026/03]** We release the InternDataEngine codebase, which includes the core modules: InternData-A1, Nimbus, and InternData-M1.

## 🚀 Quickstart

Please refer to the [Installation](TBD) and [Usage](TBD) to start the installation and run your first synthetic data generation task.

For more details, please check [Documentation](TBD).

## License and Citation
All the code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider citing our papers if it helps your research.

```BibTeX
@article{tian2025interndata,
  title={Interndata-a1: Pioneering high-fidelity synthetic data for pre-training generalist policy},
  author={Tian, Yang and Yang, Yuyin and Xie, Yiman and Cai, Zetao and Shi, Xu and Gao, Ning and Liu, Hangxu and Jiang, Xuekun and Qiu, Zherui and Yuan, Feng and others},
  journal={arXiv preprint arXiv:2511.16651},
  year={2025}
}

@article{he2026nimbus,
  title={Nimbus: A Unified Embodied Synthetic Data Generation Framework},
  author={He, Zeyu and Zhang, Yuchang and Zhou, Yuanzhen and Tao, Miao and Li, Hengjie and Tian, Yang and Zeng, Jia and Wang, Tai and Cai, Wenzhe and Chen, Yilun and others},
  journal={arXiv preprint arXiv:2601.21449},
  year={2026}
}

@article{chen2025internvla,
  title={Internvla-m1: A spatially guided vision-language-action framework for generalist robot policy},
  author={Chen, Xinyi and Chen, Yilun and Fu, Yanwei and Gao, Ning and Jia, Jiaya and Jin, Weiyang and Li, Hao and Mu, Yao and Pang, Jiangmiao and Qiao, Yu and others},
  journal={arXiv preprint arXiv:2510.13778},
  year={2025}
}
```

# RoCo: Dialectic Multi-Robot Collaboration with Large Language Models
Codebase for paper: RoCo: Dialectic Multi-Robot Collaboration with Large Language Models

[Mandi Zhao](https://mandizhao.github.io), [Shreeya Jain](https://www.linkedin.com), [Shuran Song](https://www.cs.columbia.edu/~shurans/) 

[Arxiv](https://arxiv.org/abs/2307.04738) | [Project Website](https://project-roco.github.io) 

 
<img src="method.jpeg" alt="method" width="800"/>


## Setup
### setup conda env and package install
```
conda create -n roco python=3.10
conda activate roco
```

### Install mujoco and dm_control 
```
pip install mujoco==2.3.0 dm_control==1.0.8 tokenizers==0.13.3 distro open3d ray==2.48.0
grep -v "^anthropic" requirements.txt | pip install -r /dev/stdin
pip install "openai==0.27.2" --force-reinstall
pip install -e .
```

### Install other packages
```
pip install -r requirements.txt
```

### Note on opentelemetry bugfix
```
pip install "opentelemetry-exporter-prometheus==0.58b0"
```

## Usage 
### Run multi-robot dialog on the PackGrocery Task using the latest GPT-4 model
```
$ conda activate roco
(roco) $ python run_dialog.py --task pack
```

## Contact
Please direct to [Mandi Zhao](https://mandizhao.github.io). 
If you are interested in contributing or collaborating, please feel free to reach out! I'm more than happy to chat and brainstorm together. 

## Cite
```
@misc{mandi2023roco,
      title={RoCo: Dialectic Multi-Robot Collaboration with Large Language Models}, 
      author={Zhao Mandi and Shreeya Jain and Shuran Song},
      year={2023},
      eprint={2307.04738},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
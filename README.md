# Reproduction of CheckGPT From The 2024 ACM-CCS Publication

> **Zeyan Liu, Zijun Yao, Fengjun Li, Bo Luo**  
> _ACM Computer and Communications Security (ACM-CCS), 2024_  
> [[Publication](https://dl.acm.org/doi/abs/10.1145/3658644.3670392)] | [[Video](#)] | [[Original source code](https://github.com/liuzey/CheckGPT-v2)] | [[Artifacts](https://zenodo.org/records/11215222)]

---

## Overview

This repository contains a reproduction attempt for the experiments from the ACM-CCS 2024 publication. The project focuses on constructing solutions to detect ChatGPT-generated content in academic writings in order to prevent it's abusive usage in formal publications. Such solution is needed to enforce policies introduced by academia regarding the usage of LLM-generated content. This reproduction pipeline is specialized for Chameleon Cloud.

The pipeline for the reproduction are divided into different checkpoints under the format `repro_checkpoint_X.ipynb`.

Reproduction may take up a significant amount of time, as such I recommend you use `tmux` to run the reproduction in the background like so...
1. `tmux new -s checkpoint_X`
2. `jupyter nbconvert --to notebook --execute "./repro_checkpoint_X.ipynb" --inplace --ExecutePreprocessor.timeout=0`
3. While it is running, do `Ctrl+b` then `d` to exit the tmux session. You can safely close the terminal and come back later to check if the notebook has finished running.
4. You can check the session by using `tmux attach-instance -t checkpoint_X` and see if if it's still running.
5. After the notebook has finished running all the cells, kill the session with `tmux kill-session -t checkpoint_X` (`tmux ls` should be able to confirm that the session was killed).

_Replace_ `X` _with the desired checkpoint number._

_Requires `jupyter-core` via apt and `nbconvert` via pip._

---

### System

- **Platform**: [Chameleon Cloud](https://chameleoncloud.org/)
- **GPU**: Chameleon Cloud GPU (NVIDIA Quadro RTX 6000) at CHI@UC
- **Processor**: Chameleon Cloud CPU (Intel(R) Xeon(R) Gold 6126 CPU @ 2.60GHz) at CHI@UC
- **RAM**: 192GiB
- **Image**: CC-Ubuntu22.04-CUDA
- **Workspace Kernel**: Python 3.10.12 Virtual Environment

### Results
TBA

---

## Acknowledgments

- **Computing Resources**: Chameleon Cloud (CHI@UC) for providing GPU infrastructure
- **Original Authors**: For making the training code publicly available
- **ACM-CCS 2024**: For publishing the foundational research

---

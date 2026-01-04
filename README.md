# Reproduction of CheckGPT From The 2024 ACM-CCS Publication

> **Zeyan Liu, Zijun Yao, Fengjun Li, Bo Luo**  
> _ACM Computer and Communications Security (ACM-CCS), 2024_  
> [[Publication](https://dl.acm.org/doi/abs/10.1145/3658644.3670392)] | [[Video](https://youtu.be/XUPqlPgbLT8)] | [[Original source code](https://github.com/liuzey/CheckGPT-v2)] | [[Artifacts](https://zenodo.org/records/11215222)]

---

## Overview

This repository contains a reproduction attempt for the experiments from the ACM-CCS 2024 publication. The project focuses on constructing solutions to detect ChatGPT-generated content in academic writings in order to prevent it's abusive usage in formal publications. Such solution is needed to enforce policies introduced by academia regarding the usage of LLM-generated content.

The pipeline for the reproduction is in `run_reproduction.ipynb`.

Reproduction may take up a significant amount of time, as such I recommend you use `tmux` to run the reproduction in the background like so...
1. `tmux new -s repro`
2. `jupyter nbconvert --to notebook --execute "./run_reproduction.ipynb" --inplace --ExecutePreprocessor.timeout=0`
3. While it is running, do `Ctrl+b` then `d` to exit the tmux session. You can safely close the terminal and come back later to check if the notebook has finished running.
4. You can check the session by using `tmux attach-session -t repro` and see if if it's still running.
5. After the notebook has finished running all the cells, kill the session with `tmux kill-session -t repro` (`tmux ls` should be able to confirm that the session was killed).
6. It should populate the directory `./artifact_checkgpt/CheckGPT/exp` with the experiment results under the format `Train_[TRAIN DOMAIN(S)]_[TRAIN ARCHITECTURE]_Task[TRAIN TASK(S)]_Prompt[TRAIN PROMPT(S)]_Test_[TEST DOMAIN(S)]_Task[TEST TASK(S)]_Prompt[TEST PROMPT(S)]`. Each directory contains a `train.log` file. The `summarize_reproduction.ipynb` will process these training logs into human-readable format.

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
Table 7

<img width="500" alt="Screenshot 2026-01-04 124250" src="https://github.com/user-attachments/assets/ed88a30a-ac61-4c34-8d0a-3487e2240adf" />


Table 8

<img width="500" alt="Screenshot 2026-01-04 124442" src="https://github.com/user-attachments/assets/d9fc9d50-c4bd-4056-ad2c-050a0c0843e1" />


Table 9

<img width="500" alt="Screenshot 2026-01-04 124611" src="https://github.com/user-attachments/assets/6af43b8a-357e-4310-af55-421d5a505714" />


Table 10

<img width="500" alt="Screenshot 2026-01-04 124737" src="https://github.com/user-attachments/assets/3f297f8e-b586-4d79-ba0a-cca31f5cee7d" />


---

## Acknowledgments

- **Computing Resources**: Chameleon Cloud (CHI@UC) for providing GPU infrastructure
- **Original Authors**: For making the training code publicly available
- **ACM-CCS 2024**: For publishing the foundational research

---

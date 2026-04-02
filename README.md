# ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVlabs/ToolOrchestra)

<p align="center">
  <a href="https://arxiv.org/abs/2511.21689"><img src="https://img.shields.io/badge/ArXiv-Paper-brown" alt="Paper"></a>
  <a href="https://github.com/NVlabs/ToolOrchestra/"><img src="https://img.shields.io/badge/GitHub-Code-orange" alt="Code"></a>
  <a href="https://huggingface.co/nvidia/Orchestrator-8B"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Model-green" alt="Model"></a>
  <a href="https://huggingface.co/datasets/nvidia/ToolScale"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Data-blue" alt="Data"></a>
  <a href="https://research.nvidia.com/labs/lpr/ToolOrchestra/"><img src="https://img.shields.io/badge/Project-Page-purple" alt="Website"></a>
</p>

<p align="center">
<b><a href="https://hongjin-su.github.io/">Hongjin Su</a>*</b>, <b><a href="https://shizhediao.github.io/">Shizhe Diao</a>*</b>, <a href="https://gloriaximinglu.github.io/">Ximing Lu</a>, <a href="https://research.nvidia.com/person/mingjie-liu">Mingjie Liu</a>, <a href="https://jiacheng-xu.github.io/">Jiacheng Xu</a>, <a href="https://simonxin.com/">Xin Dong</a>, <a href="https://www.yongganfu.com/">Yonggan Fu</a>, <a href="https://pbelcak.com/">Peter Belcak</a>, <a href="https://sites.google.com/site/yhrspace/home">Hanrong Ye</a>, <a href="https://hongxu-yin.github.io/">Hongxu Yin</a>, <a href="https://www.linkedin.com/in/yi-dong-04057b18/">Yi Dong</a>, <a href="https://developer.nvidia.com/blog/author/ebakhturina/">Evelina Bakhturina</a>, <a href="https://taoyds.github.io/">Tao Yu</a>, <a href="https://yejinc.github.io/">Yejin Choi</a>, <a href="https://jankautz.com/">Jan Kautz</a>, <a href="https://www.pmolchanov.com/">Pavlo Molchanov</a>
</p>

<p align="center">
<b>NVIDIA</b> &nbsp;·&nbsp; <b>The University of Hong Kong</b><br>
<sup>*</sup>Equal Contribution
</p>

---

## 📰 News

- **2025/12/5**: Our generated dataset [ToolScale](https://huggingface.co/datasets/nvidia/ToolScale) became the **#1 most-downloaded dataset** on Hugging Face, and [Nemotron-Orchestrator-8B](https://huggingface.co/nvidia/Orchestrator-8B) ranked **#3 among all models**.
- **2025/12/2**: 🏆 ToolOrchestra ranks **#1 on [GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard)**!

- **2025/11/27**: We release the code, training data, and model checkpoints of ToolOrchestra.
---

<p align="center">
<table>
<tr>
<td><img src="https://raw.githubusercontent.com/NVlabs/ToolOrchestra/main/assets/results_figure.png" alt="ToolOrchestra Performance" width="1030"/></td>
<td><img src="https://raw.githubusercontent.com/NVlabs/ToolOrchestra/main/assets/cost_performance.png" alt="Cost Performance" width="500"/></td>
</tr>
</table>
</p>

We introduce **ToolOrchestra**, a method for training small orchestrators that coordinate the use of intelligent tools. By using both tools and specialized models, ToolOrchestra surpasses GPT-5 while being much more efficient. Given a task, the Orchestrator alternates between reasoning and tool calling in multiple turns to solve it. The Orchestrator interacts with a diverse tool set, including basic tools (e.g., web search, code interpreter), specialized LLMs (e.g., coding models, math models), and generalist LLMs (e.g., GPT-5, Llama-Nemotron-Ultra-253B, Claude Opus 4.1). During training, Orchestrator is jointly optimized by outcome, efficiency, and preference rewards via end-to-end reinforcement learning. To aid RL training, we develop an automatic pipeline to synthesize both environment and tool-call tasks at scale.

<p align="center">
<img src="https://raw.githubusercontent.com/NVlabs/ToolOrchestra/main/assets/method.png" width="100%"/>
</p>

With ToolOrchestra, we produce **Orchestrator-8B**, a state-of-the-art 8B parameter orchestration model designed to solve complex, multi-turn agentic tasks by coordinating a diverse set of expert models and tools.

**Key Results:**
- On **HLE**, Orchestrator-8B achieves a score of **37.1%**, outperforming GPT-5 (35.1%) while being **2.5× more efficient**
- On **τ²-Bench** and **FRAMES**, Orchestrator-8B surpasses GPT-5 by a wide margin while using only **~30% of the cost**

---

## 🛠️ Setup Environment

```bash
# Clone this repository
git clone https://github.com/NVlabs/ToolOrchestra.git
cd ToolOrchestra

# Download index files and checkpoints
git clone https://huggingface.co/datasets/multi-train/index
export INDEX_DIR='/path/to/index'
git clone https://huggingface.co/nvidia/Nemotron-Orchestrator-8B
export CKPT_DIR='/path/to/checkpoint'

# Set environment variables
export HF_HOME="/path/to/huggingface"
export REPO_PATH="/path/to/this_repo"

# Environment for Search API, please go to [Tavily](https://app.tavily.com/home) and apply for an API key.
export TAVILY_KEY="TAVILY_KEY"
export WANDB_API_KEY="WANDB_API_KEY"
export OSS_KEY="OSS_KEY" # NVIDIA NGC key
export CLIENT_ID="CLIENT_ID"
export CLIENT_SECRET="CLIENT_SECRET"

```
# For Tau2-Bench Evaluation (Enroot Environment)

You need to re-build the image because it requires local installation of tau2:

```bash
srun --partition=interactive --time=04:00:00 --nodes=1 --overcommit --ntasks-per-node=1 --cpus-per-task=128 --job-name tau2-bench --account=nvr_lpr_agentic --gres=gpu:8  --container-image /lustre/fsw/portfolios/nvr/users/$USER/docker/s1.sqsh --container-save /lustre/fsw/portfolios/nvr/users/$USER/docker/s1.sqsh --container-mounts=$HOME:/home,/lustre:/lustre --pty /bin/bash

pip uninstall -y tau2
cd toolorchestra/evaluation/tau2-bench
pip install -e .
exit
```

### Environment for Training (Conda Environment)

```bash
conda create -n toolorchestra python=3.12 -y
conda activate toolorchestra
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
pip install -e training/rollout
```

### Environment for Retrieval

```bash
conda create -n retriever python=3.12 -y
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install transformers datasets pyserini psutil
conda install -c pytorch -c nvidia faiss-gpu
pip install uvicorn fastapi
pip install tavily-python
pip install flash-attn --no-build-isolation
```

### Environment for vLLM Models

```bash
conda create -n vllm1 python=3.12 -y
conda activate vllm1
pip install torch
pip install "transformers<4.54.0"
pip install vllm==0.9.2 # for gemma-2-9b-it
cd evaluation/tau2-bench
pip install -e .
```

---

## 🚀 Training

```bash
cd training
python resume_h100.py
```

---

## 📊 Evaluation

```bash
cd evaluation

# Evaluate on HLE (requires env: vllm1 and retriever)
python run_hle.py

# Evaluate on FRAMES (requires env: vllm1 and retriever)
python run_frames.py

# Evaluate on τ²-Bench (requires env: vllm1)
cd tau2-bench/
python run.py
```

---

## ⚙️ Customization

- **LLM Calls**: Modify the `get_llm_response` function in `LLM_CALL.py` to change LLM calls to services beyond vLLM and OpenAI
- **Prompts**: Modify lines `455-458` in `eval_hle.py` and `506-509` in `eval_frames.py`
- **Tool Configuration**: Substitute `tool_config` in line 27 of `eval_frames.py` and `eval_hle.py` for different tool sets
- **Tools & Models**: Modify `tools.json` and the `call_tool` function in `eval_hle.py`
- **Parallel Experiments**: Modify variables `{EXPERIMENT_NAME1}`, `{EXPERIMENT_NAME2}`, `{EXPERIMENT_NAME3}` in `training/resume_h100.py`, which should correspond to the file names in the directory

### Preventing Connection Errors

To prevent connection errors to host models in HLE, you may comment [this line](https://github.com/NVlabs/ToolOrchestra/blob/main/evaluation/run_hle.py#L248), then run:

```bash
# In separate processes
python run_hle.py
python eval_hle.py --model_name {cur_ckpt_dir} --output_dir {cur_output_dir} --model_config model_configs/serve2.json --example_path hle.jsonl
```

---

## 📜 License

This project is licensed under the [Apache 2.0 License](https://github.com/NVlabs/ToolOrchestra/blob/main/LICENSE).

---

## 📝 Citation

If you find this repository useful, please consider giving a ⭐ and citing our [paper](https://arxiv.org/abs/2511.21689):

```bibtex
@misc{toolorchestra,
      title={ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration}, 
      author={Hongjin Su and Shizhe Diao and Ximing Lu and Mingjie Liu and Jiacheng Xu and Xin Dong and Yonggan Fu and Peter Belcak and Hanrong Ye and Hongxu Yin and Yi Dong and Evelina Bakhturina and Tao Yu and Yejin Choi and Jan Kautz and Pavlo Molchanov},
      year={2025},
      eprint={2511.21689},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.21689}, 
}
```

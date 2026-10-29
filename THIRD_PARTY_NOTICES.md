# Third-party notices

MakeMarketBench code is MIT, copyright 2026 Nicolas Coquelet. That licence does
not relicense model weights, hosted services, or their outputs. No third-party
model weights or training frameworks are vendored in this repository.

Model-card licence fields and immutable Hub revisions were retrieved on
2026-10-29; see [the verification record](results/licenses.json). A Hub revision
does **not** identify the unpublished backend revision served by NIM.

## Models

| Model / conversion | Exact licence and primary source | Attribution / handling |
|---|---|---|
| Qwen3-1.7B; `mlx-community/Qwen3-1.7B-4bit` | [Apache-2.0](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE) | Qwen Team, Alibaba Cloud; MLX conversion by MLX Community. Retain the licence, applicable notices, and mark modifications when redistributing weights/adapters. The [full licence](third_party/qwen3_LICENSE.txt) and a modification notice accompany the private checkpoint dataset ([receipt](results/training/kaggle_dataset_notice.json)). |
| Qwen3.5-4B; `mlx-community/Qwen3.5-4B-MLX-4bit`; `unsloth/Qwen3.5-4B` | [Apache-2.0](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE) | Qwen Team, Alibaba Cloud; MLX Community and Unsloth conversions. Same retention/changed-file obligations. The Unsloth checkpoint is used by the executed CUDA smoke and full training job. |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | [Apache-2.0 model card](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit) | Optional memory-dependent candidate; not an executed result unless recorded. |
| Qwen3.8-27B; Unsloth NF4 derivative | [Apache-2.0 original](https://huggingface.co/Qwen/Qwen3.8-27B); [Apache-2.0 derivative](https://huggingface.co/unsloth/Qwen3.8-27B-unsloth-bnb-4bit) | Two-T4 LoRA SFT/GRPO fit verified; selected for training. Preserve Apache licence, attribution and modification notices when redistributing. Pinned revisions and weight metadata: `results/training/qwen38_quantized.json`. |
| `openai/gpt-oss-20b` | [Apache-2.0 model card](https://huggingface.co/openai/gpt-oss-20b) | OpenAI. API inference only; no weights redistributed. |
| NVIDIA Nemotron-3-Ultra-550B-A55B; Nemotron-3.5-Lightning-30B-A3B; Nemotron-3.5-Bolt-E4B | [OpenMDW License Agreement 1.1](https://openmdw.ai/license/1-1/), verified through the [Ultra](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16), [Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16), and [Bolt](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) model cards | NVIDIA Corporation. Redistributed model materials require the agreement and applicable origin/copyright notices. API inference only here. |
| `zai-org/GLM-5.3-Flash` | [MIT model card](https://huggingface.co/zai-org/GLM-5.3-Flash) | Z.ai; retain copyright and licence if redistributing model materials. Probe outcomes are recorded separately. |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | [MIT model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | DeepSeek; retain copyright and licence if redistributing model materials. This is a licence reference, not proof of NIM's exact deployment revision. |
| Llama-3.3-70B-Instruct | [Llama 3.3 Community License Agreement](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE) | Meta Platforms, Inc. Custom terms, including required agreement/NOTICE retention and “Built with Llama” attribution for covered redistribution/products, acceptable-use terms, and the large-user commercial provision. API candidate only unless an evaluation is recorded. |
| Llama-3.1-8B-Instruct | [Llama 3.1 Community License Agreement](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) | Meta Platforms, Inc.; retain the version-specific agreement, origin notice and required “Built with Llama” attribution for covered products/distribution. API candidate only. |
| MiniMax-M3.1-Flash | [MiniMax Community License](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/LICENSE) | Copyright 2026 MiniMax. **Default grant is non-commercial.** Commercial use has additional attribution and notice/authorisation conditions. Historical format/availability probes only; no weights or training labels used. |
| Qwen3-235B-A22B | [Apache-2.0 model card](https://huggingface.co/Qwen/Qwen3-235B-A22B) | Qwen Team; API candidate, not redistributed. |
| Mistral-Small-3.1-24B-Instruct-2503 | [Apache-2.0 model card](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) | Mistral AI; API candidate, not redistributed. |
| Mistral-7B-Instruct-v0.3; Mistral-Nemo-Instruct-2407 | Apache-2.0: [7B model card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), [NeMo model card](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) | Mistral AI / NVIDIA. Retain licences and notices for covered redistribution; API inference only. |
| Llama-3.2-11B-Vision-Instruct | [Llama 3.2 Community License Agreement](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) and [model-card Acceptable Use Policy](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) | Meta Platforms, Inc. Retain agreement, origin notice and required “Built with Llama” attribution for covered products/distribution. The multimodal EU restriction has an end-user exception for products/services incorporating the model; this repo only calls the hosted NIM API with text, and does not download or redistribute these weights. |

Model availability probes are not evaluations. Models that return 404 or never
produce a usable quote have no leaderboard score. Other historical unsuccessful
probes in `results/endpoint_check.json` did not download model materials.

## Libraries and tools

These are installed dependencies, not copied implementations. Follow their
licences when redistributing environments or binaries; this table is not a
replacement for their complete notices and dependency-specific terms.

| Component | Licence / upstream | Required handling for redistribution |
|---|---|---|
| MLX and MLX-LM | MIT: [MLX](https://github.com/ml-explore/mlx/blob/main/LICENSE), [MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE) | Retain Apple Inc. copyright, permission notice and disclaimer. Training calls the upstream trainer and LoRA implementations. |
| MLX-LM chunkwise gated-delta port | [MIT text](third_party/mlx_lm_LICENSE.txt), copyright Apple Inc. | Two algorithm functions from adityak74's [PR #1870](https://github.com/ml-explore/mlx-lm/pull/1870), pinned to `3b04b9f1992fb83c4210228f7eb1c9dc72827e49`. Type annotations/import style changed; algorithm bodies unchanged. Source and licence hashes are in `results/licenses.json`. |
| Unsloth Core | [Apache-2.0](https://github.com/unslothai/unsloth/blob/main/LICENSE) | Retain licence/NOTICE and mark changes. The optional AGPL-3.0 Studio/Docker UI is not used or bundled. |
| Unsloth Zoo | [GNU LGPL version 3](https://github.com/unslothai/unsloth-zoo/blob/main/LICENSE) | Executed CUDA runtime dependency; not vendored. Redistribution must retain its LGPL/GPL texts and library-source/relinking rights as applicable. It must not be labelled Apache-2.0 merely because Unsloth Core is Apache-2.0. |
| pybind11 | [BSD-3-Clause](https://github.com/pybind/pybind11/blob/master/LICENSE) | Retain Wenzel Jakob/contributors copyright, conditions and disclaimer; no endorsement. |
| NumPy | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | Retain NumPy Developers copyright, conditions and disclaimer. |
| PyTorch | [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) | Retain upstream copyright, licence and bundled third-party notices. CUDA training dependency. |
| Transformers, TRL, PEFT, Datasets, Hugging Face Hub, Safetensors | Apache-2.0: [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE), [TRL](https://github.com/huggingface/trl/blob/main/LICENSE), [PEFT](https://github.com/huggingface/peft/blob/main/LICENSE), [Datasets](https://github.com/huggingface/datasets/blob/main/LICENSE), [Hub](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE), [Safetensors](https://github.com/huggingface/safetensors/blob/main/LICENSE) | Retain respective licences/NOTICE files and identify modifications. |
| OpenAI Python client | [Apache-2.0](https://github.com/openai/openai-python/blob/main/LICENSE) | Retain licence/NOTICE; used as an OpenAI-compatible NIM client. |
| Matplotlib | [Matplotlib licence based on PSF](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) | Retain licence, copyright and derivative-change summary where required. |
| scikit-build-core; pytest; Ruff; Kaggle CLI | [Apache-2.0](https://github.com/scikit-build/scikit-build-core/blob/main/LICENSE); [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE); [MIT](https://github.com/astral-sh/ruff/blob/main/LICENSE); [Apache-2.0](https://github.com/Kaggle/kaggle-cli/blob/main/LICENSE.txt) | Retain applicable licences and notices when redistributing these tools. |

## Data, market rules and hosted services

Algorithm attribution: the ID map's xor/multiply/xor mix uses the first part
of Austin Appleby's [MurmurHash3 `fmix64`](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp).
His upstream header places that source in the public domain and disclaims
copyright; this is a shortened variant, not the full MurmurHash3 algorithm.
The replay checksum uses an FNV-style word mix with a repository-specific seed.
Glenn Fowler, Landon Curt Noll and Phong Vo dedicated
[FNV](https://www.isthe.com/chongo/tech/comp/fnv/index.html) to **CC0-1.0**;
no attribution is required, but the algorithm's origin is acknowledged here.

- **Synthetic games and demonstrations:** generated by this repository from
  deterministic seeds and the exact Bayes teacher. No exchange, Binance, Yahoo,
  broker, personal, or proprietary financial dataset is used. Generation:
  `python train/make_sft_data.py --episodes 800`. Original synthetic data is MIT
  under Nicolas Coquelet's copyright; model derivatives retain model terms.
- **Figgie-lite:** an original simplified implementation inspired by
  [Jane Street's Figgie](https://www.figgie.com/). No Jane Street assets, site
  code, or third-party bot code is copied. The game name is acknowledged; no
  affiliation or endorsement is implied.
- **NVIDIA hosted inference:** the [API Trial Terms of Service](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf)
  apply separately from weight licences. Sections 1.2/1.4 limit trial service and
  generated content to testing/evaluation; section 4.2 restricts redistribution.
  Cached answers and historical raw transcripts are retained in this private
  research repository for audit. **Public redistribution of those transcripts
  requires the owner's terms review/appropriate rights; the code's MIT licence
  does not grant that permission.** No API output is used as training targets:
  all expert labels are computed locally by the Bayes dealer.
- **Kaggle:** [Kaggle terms](https://www.kaggle.com/terms) govern the hosted
  execution service. Kernel metadata uses `is_private: true`; dataset creation
  omits `--public`. No credentials are embedded. Authentication failure is
  recorded, not described as a GPU run.

No CC-BY-NC dataset is used. MiniMax has a non-commercial default grant with
conditional commercial permissions. NIM trial usage and Llama also have custom
restrictions; these materials must not be described as unrestricted MIT software.

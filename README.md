# RLHF Pipeline for Small Open-Source LLMs (GPT-2 vs DistilGPT2)

A hands-on implementation and evaluation of the three-stage **Reinforcement Learning from Human Feedback (RLHF)** pipeline (SFT → reward modelling → PPO) on small open-source language models.

Built for *ARTI 6000: Advanced Topics in AI & Machine Learning* (University of Adelaide, Assignment 1).

**Research question:** Does model capacity significantly impact the effectiveness of RLHF in small, open-source language models?

---

## Pipeline

```
Base LLM ──► 1. Supervised Fine-Tuning ──► 2. Reward Model ──► 3. PPO (RL fine-tuning) ──► Evaluation
              (instruction/text data)       (scores responses)    (reward − KL penalty)     (reward score + samples)
```

1. **SFT:** adapt the base model to the target text format.
2. **Reward modelling:** train a model that outputs a scalar score for a response.
3. **PPO:** generate responses, score them with the frozen reward model, and update the policy with a clipped PPO objective plus a KL penalty against the SFT model to stop it drifting.

## Models and datasets

| Stage | GPT-2 (124M, 12 layers) | DistilGPT2 (82M, 6 layers) |
|---|---|---|
| SFT data | SST-2 text (`stanfordnlp/sst2`) | `yahma/alpaca-cleaned` (1% subset) |
| Reward model | GPT-2 + linear reward head, trained on SST-2 labels as a proxy preference signal | `AutoModelForSequenceClassification` trained with TRL `RewardTrainer` on a tiny hand-made preference set (2 pairs) |
| PPO prompts | SST-2 sentences with more than 8 words (31,105 train / 807 validation), truncated to 2–7 tokens | SST-2 / Alpaca tokenised data via TRL experimental PPO |

An additional notebook runs **Falcon-7B-Instruct** (`tiiuae/falcon-7b-instruct`) as an inference baseline on a Tesla T4, with exploratory reward-model and DPO cells on a 1,000-example subset of Anthropic **HH-RLHF**.

**Tools:** Python, PyTorch, Hugging Face Transformers / Datasets, TRL, Google Colab (single T4 GPU).

## Repository contents

| File | What it does |
|---|---|
| `1_SFT_GPT2.ipynb` | Supervised fine-tuning of GPT-2 |
| `2_RM_Training_GPT2.ipynb` | Reward model (GPT-2 + reward head), validation loss and confusion matrix |
| `3_RLHF_PPO_GPT2.ipynb` | PPO from scratch: value head, KL-penalised reward, advantage estimation, clipped loss, validation |
| `RLHF_Distillgpt2.ipynb` | Full SFT → RM → PPO run for DistilGPT2 using TRL |
| `RLHF_LLM_tiiuae_falcon_7b_instruct.ipynb` | Falcon-7B-Instruct baseline and exploratory HH-RLHF experiments |
| `RLHF_Assignment1.pdf` | Written report |
| `RLHF_LLMs.pptx` | Presentation slides |
| `RLHF_video.mp4` | Recorded presentation (about 10 minutes) |

## How to run

1. Open the notebooks in **Google Colab** and select a GPU runtime (T4 is enough).
2. Run the notebooks in order: `1_SFT_GPT2` → `2_RM_Training_GPT2` → `3_RLHF_PPO_GPT2`. Each notebook saves its model to Google Drive, and the next one loads it, so mount Drive when prompted.
3. `RLHF_Distillgpt2.ipynb` and the Falcon notebook are standalone.
4. Install dependencies with `pip install transformers datasets accelerate trl peft bitsandbytes evaluate`. TRL's PPO API changes between versions, so if the DistilGPT2 notebook fails, check the TRL version.

## Results

### GPT-2: PPO improves the reward

| Metric | Value |
|---|---|
| Reward model validation loss | 0.439 |
| Reward model validation accuracy (confusion matrix) | ~79% (688 / 867 correct) |
| Average reward on 807 validation prompts, SFT model | 0.0648 |
| Average reward on 807 validation prompts, after PPO | **0.1185** (~83% increase) |

### DistilGPT2: PPO degrades the model

After PPO, DistilGPT2 produced incoherent, off-topic text (the policy collapsed), even though SFT had worked.

| Model | Stage | Sample output |
|---|---|---|
| DistilGPT2 | SFT | "Gen AI is a system that learns and improves." |
| DistilGPT2 | PPO | Drifts into unrelated text about politics and courts |
| GPT-2 | SFT | "for young women who find themselves of color for many during the years" |
| GPT-2 | PPO | "is surprisingly watchable, though it is not for everyone" |

### Conclusion

Model capacity matters. GPT-2 (124M, 12 layers) absorbed the reward signal and produced more coherent, higher-scoring text. DistilGPT2 (82M, 6 layers, higher perplexity: 21.1 vs 16.3 on WikiText-103) did not.

## Limitations

- **Tiny reward data.** The reward models were trained on a proxy signal (SST-2 sentiment labels) and a 2-pair hand-made set, not a real human preference dataset such as HH-RLHF. Treat this as a proof of concept of the pipeline.
- **Proxy evaluation.** Results rely on reward-model scores and a handful of samples, not blind human comparisons. A reward increase can also reflect reward hacking.
- **Small models and compute.** Everything ran on a single T4, so only small models and a partial SFT pass were possible. The GPT-2 SFT run was stopped before the epoch finished.
- **Single run.** No repeated seeds or confidence intervals.

## Future work

- Scale to 7B models (Llama 2, Mistral) with LoRA/QLoRA.
- Train reward models on real preference data (HH-RLHF, OpenAssistant, HelpSteer2).
- Compare PPO against DPO.
- Use TRL or OpenRLHF throughout, and add human or LLM-as-judge evaluation.

## Acknowledgements

The from-scratch PPO implementation was adapted from [ash80/RLHF_in_notebooks](https://github.com/ash80/RLHF_in_notebooks). Methods follow InstructGPT (Ouyang et al., 2022), HH-RLHF (Bai et al., 2022) and DPO (Rafailov et al., 2023).

## Author

Jaden Machado · [GitHub](https://github.com/Jaden449) · [LinkedIn](https://www.linkedin.com/in/jaden-machado-76ba33283)


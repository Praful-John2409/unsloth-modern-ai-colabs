# unsloth-modern-ai-colabs

A suite of **five end-to-end Colab notebooks** showcasing modern AI workflows with **Unsloth.ai**: full fine-tuning, LoRA/PEFT, preference-based **RL** (PPO/RLHF-style), **GRPO reasoning**, and **continued pretraining** (teach a model a new language/domain).
Each notebook is designed to **run to completion**, produce **exportable checkpoints**, and includes a **YouTube walkthrough** script + checklist.

> **Submission**: push this repo to GitHub and add your video links in `docs/VIDEOS.md`. Each Colab must run successfully and save artifacts to Drive/HF Hub.

---

## 🗂️ Repository Structure

```
unsloth-modern-ai-colabs/
├─ README.md
├─ LICENSE
├─ notebooks/
│  ├─ 01_full_ft_smollm2_135m.ipynb
│  ├─ 02_lora_smollm2_135m.ipynb
│  ├─ 03_rl_preference_pairs.ipynb
│  ├─ 04_grpo_reasoning.ipynb
│  └─ 05_continued_pretraining_new_language.ipynb
├─ datasets/
│  ├─ samples/
│  │  ├─ chat_alpaca_10.jsonl
│  │  ├─ code_instruct_10.jsonl
│  │  ├─ pref_pairs_8.jsonl        # (prompt, chosen, rejected)
│  │  ├─ grpo_math_tiny.jsonl      # (problem, solution)
│  │  └─ cpt_new_lang_tiny.jsonl   # (raw text lines)
│  └─ README.md
├─ docs/
│  ├─ VIDEOS.md
│  ├─ DATA_FORMATS.md
│  ├─ EVAL.md
│  ├─ COSTING.md
│  └─ TROUBLESHOOTING.md
└─ export/
   ├─ ollama/
   └─ hf_model_cards/
```

---

## 🧰 Prerequisites (Colab-friendly)

* Google Colab (T4/L4/A100 where available)
* 🤗 **transformers**, **trl**, **datasets**, **accelerate**
* **unsloth** library (installed in notebooks)
* Optional: **Hugging Face** account + token (to push checkpoints)
* Optional: **Kaggle** datasets access (for larger corpora)

---

## 📹 Video Requirements (for each notebook)

Include in your walkthrough:

1. **Goal & model choice** (why this model, task alignment)
2. **Dataset & format** (show a JSONL row)
3. **Code walk** (key cells, hyperparams, training loop)
4. **Live training snippet** (few steps) & **logs**
5. **Evaluation** (quick metrics or qualitative samples)
6. **Export** (HF Hub / GGUF / Ollama) + **inference demo**
7. **Takeaways & pitfalls**

Add links to `docs/VIDEOS.md`.

---

## 🔎 Data Formats (quick view)

See `docs/DATA_FORMATS.md` for details.

**Supervised chat/code** (`*.jsonl`)

````json
{"system":"You are a helpful assistant.","input":"Write a Python function for factorial.","output":"```python\ndef fact(n): ...\n```"}
````

**Preference pairs for RL** (`pref_pairs_*.jsonl`)

```json
{"prompt":"Explain REST vs gRPC.",
 "chosen":"Concise, accurate explanation with tradeoffs.",
 "rejected":"Vague and incorrect answer."}
```

**GRPO reasoning** (`grpo_*.jsonl`)

```json
{"question":"If 3x+2=11, find x.","answer":"x = 3"}
```

**Continued pretraining** (`cpt_new_lang_*.jsonl`)

```json
{"text":"<new-language raw sentence here>"}
```

---

## 🧪 Evaluation (lightweight)

See `docs/EVAL.md`:

* **SFT/LoRA**: instruction-following exact-match on mini tasks + qualitative chat/code tests
* **RL (pref)**: reward model/length penalties; win-rate vs base on a small eval set
* **GRPO**: math/logic accuracy on mini GSM8K-style subset
* **Continued pretraining**: perplexity drop on held-out corpus + qualitative generations

---

## 💻 The Five Colabs

### 1) Full Fine-Tuning (SFT) — **smollm2-135M** (or Gemma 3 1B Unsloth 4-bit)

**Notebook**: `01_full_ft_smollm2_135m.ipynb`
**Goal**: Demonstrate **full_finetuning=True** on a tiny model for chat or code.

* **Base model options**:

  * `unsloth/smollm2-135m` (tiny, fast)
  * `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` (QLoRA-ready but we’ll still show full FT where feasible)
* **Task**: choose **chat** or **coding**
* **Key settings**:

  * `full_finetuning=True`
  * seq len 1024, lr ~ `2e-4` (tiny models), batch 64 (with grad accum), epochs 3–5
  * optimizer: AdamW8bit or AdamW
* **Outputs**:

  * saved adapter/full weights to Drive
  * sample generations (5 prompts)
  * optional: push to HF Hub
* **Video**: explain **why full FT** (small models converge quickly; baseline for LoRA comparison)

### 2) **LoRA/PEFT** — **smollm2-135M** (same dataset & eval)

**Notebook**: `02_lora_smollm2_135m.ipynb`
**Goal**: Repeat (1) with **LoRA** for parameter-efficient FT; compare runtime/quality.

* **Key settings**:

  * `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`, target modules per Unsloth docs
  * `bnb_4bit=True` when using larger bases (demonstrate best practice)
* **Compare**:

  * wall-clock time, VRAM usage, sample quality vs full FT
* **Export**:

  * Merge LoRA → full weights (optional)
  * **Ollama** export (template provided) & quick local inference demo
* **Video**: highlight **when LoRA is preferred** (bigger models, lower cost, fast iterate)

### 3) **RL with Preference Data** (PPO/RLHFlow-style)

**Notebook**: `03_rl_preference_pairs.ipynb`
**Goal**: Use a dataset with **(prompt, chosen, rejected)** to train with Unsloth RL.

* **Data**: tiny subset of open preference pairs (or curated samples in `datasets/samples/pref_pairs_8.jsonl`)
* **Flow**:

  1. Load SFT base (from #2 merged weights or base model)
  2. Prepare RM or use implicit reward from pairs
  3. Train RL (short run) with KL penalty
  4. Evaluate win-rate on held-out pairs
* **Key params**:

  * `beta_kl≈0.02–0.1`, `ppo_epochs=1–3`, `rollout_steps=64–256`
* **Video**: explain **why RL** improves helpfulness/harmlessness vs pure SFT

### 4) **GRPO Reasoning** (train your own R1-style)

**Notebook**: `04_grpo_reasoning.ipynb`
**Goal**: Use **GRPO** to improve **step-by-step reasoning** on small math/logic tasks.

* **Data**: tiny math reasoning set (`grpo_math_tiny.jsonl`), or mini-GSM8K style
* **Flow**:

  1. Initialize from SFT/LoRA base
  2. Enable **reasoning traces** (chain-of-thought hidden at inference, but used in training)
  3. Train GRPO; monitor accuracy
  4. Validate with self-consistency (few samples)
* **Key params**:

  * steps 100–500 (short demo), entropy coef small, temperature/num_samples for SC
* **Video**: connect GRPO to **rewarding intermediate steps** and stabilizing reasoning

### 5) **Continued Pretraining** (teach a new language/domain)

**Notebook**: `05_continued_pretraining_new_language.ipynb`
**Goal**: Domain/language adaptation via **unsupervised next-token** training.

* **Data**: small curated corpus in target language or domain (`cpt_new_lang_tiny.jsonl`)
* **Flow**:

  1. Tokenizer check (new vocab vs BPE coverage)
  2. Pack sequences; train with LM loss
  3. Measure **perplexity** drop on held-out text
  4. Qualitative generations before/after
* **Key params**:

  * lr ~ `5e-5`, steps 500–2k (demo), seq len 1024–2048, weight decay 0.1
* **Video**: clarify **continued pretraining vs SFT** and where each shines

---

## 🛠️ Optional Extensions (extra credit)

* **Start from a custom checkpoint** (resume training): show how to load and continue
* **Mental-health chatbot SFT** (use tiny, **non-diagnostic** dataset; add **safety disclaimer**)
* **Export to Ollama** and run local inference (`docs/TROUBLESHOOTING.md` has tips)

---

## 🔐 Safety & Ethics

* Use **synthetic** or **public non-PII** datasets.
* Mental-health notebook must include a **clear disclaimer**: not a medical device, informational only, seek professional help for crisis scenarios.
* Do not train on copyrighted/private data.

---

## 📦 Artifacts to Provide

* ✅ The 5 Colabs, each **executed successfully**
* ✅ Saved checkpoints (Drive or HF Hub links)
* ✅ Sample generations & evaluation logs
* ✅ **YouTube** links added to `docs/VIDEOS.md`
* ✅ (If exported) **Ollama** `Modelfile` + quick run command

---

## 🧪 Quick Inference Snippets (after training)

**Transformers**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("your-hf/repo", device_map="auto")
t = AutoTokenizer.from_pretrained("your-hf/repo")
x = t.apply_chat_template([{"role":"user","content":"Write a unit test for add(a,b)."}], return_tensors="pt")
y = m.generate(x.to(m.device), max_new_tokens=200)
print(t.decode(y[0], skip_special_tokens=True))
```

**Ollama (after export)**

```bash
ollama create my-unsloth-model -f export/ollama/Modelfile
ollama run my-unsloth-model "Explain binary search in 3 steps."
```

---

## 🧾 Costing & Runtime (ballpark)

See `docs/COSTING.md`. Short-run demos on T4/L4 typically finish in **10–40 minutes** per notebook with tiny datasets; GRPO can take longer. Prefer **LoRA** for bigger bases.

---

## 🆘 Troubleshooting

Common issues and fixes are in `docs/TROUBLESHOOTING.md`:

* CUDA out of memory → reduce seq length / batch / enable 4-bit
* Tokenizer mismatch → re-create tokenizer or use `trust_remote_code`
* Diverging loss → warmup steps, lower LR, gradient clipping

---

## 🙏 Credits & Hints

* Unsloth notebooks & guides:

  * [https://github.com/unslothai/notebooks/](https://github.com/unslothai/notebooks/)
  * [https://docs.unsloth.ai/get-started/fine-tuning-llms-guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
  * [https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)
  * [https://docs.unsloth.ai/basics/continued-pretraining](https://docs.unsloth.ai/basics/continued-pretraining)
* Kaggle tutorial: [https://www.kaggle.com/code/kingabzpro/fine-tuning-llms-using-unsloth](https://www.kaggle.com/code/kingabzpro/fine-tuning-llms-using-unsloth)
* LoRA + Ollama write-up (good talking points for the video)

---

> **Repo name suggestion**: `unsloth-modern-ai-colabs`.
> If you already have a portfolio monorepo, place this under `/unsloth/` and keep the notebook names unchanged for grading.

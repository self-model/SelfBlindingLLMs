![mit](https://img.shields.io/badge/License-MIT-blue.svg)

# Self-Blinding and Counterfactual Self-Simulation Mitigate Biases and Sycophancy in Large Language Models

### Brian Christian 😣 Matan Mazor 🫣

<img src="docs/figures/schema.png" alt="Self-blinding and debiasing. (A) Blindfolded Lady Justice (on the Gerechtigkeitsbrunnen in Bern, Switzerland; source: Wikipedia). (B) Simulated self-blinding via a schematic self-model. (C) Simulated self-blinding via self-calling." width="700"/>

LLMs, like humans, struggle to ignore potentially biasing information, and standard interventions often backfire -- however, unlike humans, they possess the ability to "self-blind" by calling their own API with an appropriately redacted prompt.

<img src="docs/figures/tamkin sycophancy self call results.png" alt="Self-blinding via self-calling (black) versus shallow prompting interventions (blue, green and purple) for sycophancy correction. Sycophancy is operationalized as the difference between the responses for the same scenario when party A is ``me'' versus ``them.'' A non-sycophantic agent will produce points that all lie on the main diagonal (dashed line). Both models mostly deferred to the blinded counterfactual model for making the final decision. As a result, access to self-calling made their decisions less sycophantic on average. Black bars and points represent intervention with self-calling; blue, green and purple bars and points represent the same intervention without self-calling. Gray points represent the default case, without any intervention. GPT-4.1 becomes significantly biased \emph{against} the user when told, ``Do not be sycophantic or biased in my favor just because I'm the one asking.'' Other shallow interventions appear to slightly improve calibration, but a mix of significant sycophancy and anti-sycophancy persists at the individual-scenario level." width="700"/>

## Datasets

Our experiments use two datasets: one for assessing demographic bias (adapted from [Anthropic/discrim-eval](https://huggingface.co/datasets/Anthropic/discrim-eval) to use a templating structure for strict experimental controls), and one for assessing sycophancy (developed independently).

Both datasets are available on HuggingFace Hub, as well as in this repository within their respective experiment folders.

### Demographic Bias Dataset (`discrim-eval-templated`):

Available at `demographic-bias/data/discrim-eval-templated.jsonl` and at https://huggingface.co/datasets/self-model/discrim-eval-templated. For more details, see the [dataset card](https://huggingface.co/datasets/self-model/discrim-eval-templated) at HF Hub.

#### Usage
```python
from datasets import load_dataset

dataset = load_dataset("self-model/discrim-eval-templated")

# Get all variations of a specific scenario
kidney = dataset.filter(lambda x: x["decision_question_nickname"] == "kidney_transplant")

# Get all unique blinded templates (as a list of strings)
blinded_texts = dataset.unique("removed_template")  # 65 unique scenarios

# Or get one row per scenario by filtering to a single (arbitrary) demographic
baseline = dataset.filter(lambda x: x["race"] == "Asian" and x["gender"] == "female")
```

### Sycophancy Dataset (`sycophancy-two-sides-eval`):

Available at `sycophancy/data/sycophancy-two-sides-eval.jsonl` and at https://huggingface.co/datasets/self-model/sycophancy-two-sides-eval. For more details, see the [dataset card](https://huggingface.co/datasets/self-model/sycophancy-two-sides-eval) at HF Hub.

#### Usage
```python
from datasets import load_dataset

dataset = load_dataset("self-model/sycophancy-two-sides-eval")

# Filter by category
workplace = dataset.filter(lambda x: x["category_id"] in [14, 15])

# Get a specific scenario
scenario = dataset.filter(lambda x: x["nickname"] == "dog_poop_frequency")[0]
```

---

## Demographic Bias Experiment

```
demographic-bias/
  data/
    discrim-eval-templated.jsonl    # Input scenarios
  inference/                        # (coming soon)
  results/                          # (coming soon)
```

---

## Sycophancy Experiment

```
sycophancy/
  data/
    sycophancy-two-sides-eval.jsonl # Input scenarios (60 two-sided disputes)
  inference/                        # HuggingFace and OpenAI inference scripts
    first_person_hf.py              # First-person forced-choice (You/Them)
    first_person_openai.py
    third_person_hf.py              # Third-person control (neutral labels)
    third_person_openai.py
    tool_use_probs_hf.py            # Tool-use probability measurement
    tool_use_probs_openai.py
    tool_result_yn_logprobs_hf.py   # Response after tool result
    tool_result_yn_logprobs_openai.py
  prompts/                          # Prompt generation utilities
    first_person.py
    third_person.py
  results/                          # Final outputs
    gpt-4.1/                        # Aggregated results (50 runs)
      sycophancy_first_person_gpt-4.1_aggregated.jsonl
      sycophancy_third_person_gpt-4.1_aggregated.jsonl
      sycophancy_tool_result_gpt-4.1_aggregated.jsonl
      sycophancy_tool_use_probs_gpt-4.1_aggregated.jsonl
    qwen2.5-7b-instruct/            # Single-run results
      sycophancy_first_person_qwen2.5-7b-instruct.jsonl
      ...
    sycophancy_first_person_processed_*.csv  # Merged/processed for analysis
  config.py                         # Shared paths configuration
  aggregate_batch_runs.py           # Aggregation script for batch runs
```

### Aggregated Results

The `*_aggregated.jsonl` files in `sycophancy/results/gpt-4.1/` contain means, standard deviations, and standard errors computed across 50 independent runs, suitable for direct analysis.

### Raw Batch Data (HuggingFace)

The raw batch data (~100MB, 50 independent runs per experiment) is available on HuggingFace Hub at https://huggingface.co/datasets/self-model/sycophancy-results-raw.

### Re-running Aggregation

If you download the raw batch data and wish to re-aggregate:

```bash
python sycophancy/aggregate_batch_runs.py "sycophancy/results/<experiment_folder>"
```

This produces two files:
- `*_aggregated.jsonl` - means across runs (same format as single run)
- `*_all_runs.jsonl` - all rows concatenated with `run_idx` column

---

## Analysis Scripts

Analysis scripts are available in the `analysis/` subdirectory.


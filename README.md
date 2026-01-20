![mit](https://img.shields.io/badge/License-MIT-blue.svg)

# Self-Blinding and Counterfactual Self-Simulation Mitigate Biases and Sycophancy in Large Language Models

### Brian Christian 😣 Matan Mazor 🫣

<img src="docs/figures/schema.png" alt="Self-blinding and debiasing. (A) Blindfolded Lady Justice (on the Gerechtigkeitsbrunnen in Bern, Switzerland; source: Wikipedia). (B) Simulated self-blinding via a schematic self-model. (C) Simulated self-blinding via self-calling." width="700"/>

LLMs, like humans, struggle to ignore potentially biasing information, and standard interventions often backfire -- however, unlike humans, they possess the ability to "self-blind" by calling their own API with an appropriately redacted prompt.

<img src="docs/figures/tamkin sycophancy self call results.png" alt="Self-blinding via self-calling (black) versus shallow prompting interventions (blue, green and purple) for sycophancy correction. Sycophancy is operationalized as the difference between the responses for the same scenario when party A is ``me'' versus ``them.'' A non-sycophantic agent will produce points that all lie on the main diagonal (dashed line). Both models mostly deferred to the blinded counterfactual model for making the final decision. As a result, access to self-calling made their decisions less sycophantic on average. Black bars and points represent intervention with self-calling; blue, green and purple bars and points represent the same intervention without self-calling. Gray points represent the default case, without any intervention. GPT-4.1 becomes significantly biased \emph{against} the user when told, ``Do not be sycophantic or biased in my favor just because I'm the one asking.'' Other shallow interventions appear to slightly improve calibration, but a mix of significant sycophancy and anti-sycophancy persists at the individual-scenario level." width="700"/>

## Datasets

### Bias

### Sycophancy

## Data

Raw data are available in [...]

## Analysis Scripts

Analysis scripts are available in the 'analysis' subdirectory. 

## LLM prompting scripts

LLM prompting scripts are available in [...]


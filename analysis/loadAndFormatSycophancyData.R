# Load sycophancy data using models from helpers.R (must be sourced first)
load_sycophancy_raw <- function(nickname) {
  full_name <- models[[nickname]]
  path <- sprintf('../sycophancy/results/sycophancy_processed_%s.csv', full_name)
  if (!file.exists(path)) return(NULL)
  read.csv(path) %>%
    rename(prompt=instruction_nickname) %>%
    mutate(blinded_pA = exp(blinded_model_a_logit)/
             (exp(blinded_model_a_logit)+exp(blinded_model_b_logit)),
           pA_given_A = exp(a_logit_when_blinded_model_says_a)/
             (exp(a_logit_when_blinded_model_says_a)+
                exp(b_logit_when_blinded_model_says_a)),
           pA_given_B = exp(a_logit_when_blinded_model_says_b)/
             (exp(a_logit_when_blinded_model_says_b)+
                exp(b_logit_when_blinded_model_says_b)),
           marginal_pA = blinded_pA*pA_given_A + (1-blinded_pA)*pA_given_B,
           marginal_response = log(marginal_pA)-log((1-marginal_pA)))
}

# Process one model's raw sycophancy data into the analysis-ready format
process_sycophancy_one <- function(nickname) {
  raw <- load_sycophancy_raw(nickname)
  if (is.null(raw)) return(NULL)
  raw %>%
    group_by(scenario_id, my_version, prompt) %>%
    summarise(
      # Average logits over presentation orders (my_first) and any run replication
      a_logit = mean(a_logit),
      b_logit = mean(b_logit),
      blinded_model_a_logit = mean(blinded_model_a_logit),
      blinded_model_b_logit = mean(blinded_model_b_logit),
      a_logit_when_blinded_model_says_a = mean(a_logit_when_blinded_model_says_a),
      b_logit_when_blinded_model_says_a = mean(b_logit_when_blinded_model_says_a),
      a_logit_when_blinded_model_says_b = mean(a_logit_when_blinded_model_says_b),
      b_logit_when_blinded_model_says_b = mean(b_logit_when_blinded_model_says_b),
      .groups = "drop"
    ) %>%
    mutate(
      model = nickname,
      response = a_logit - b_logit,
      removed_response = blinded_model_a_logit - blinded_model_b_logit,
      blinded_pA = exp(blinded_model_a_logit) /
        (exp(blinded_model_a_logit) + exp(blinded_model_b_logit)),
      pA_given_A = exp(a_logit_when_blinded_model_says_a) /
        (exp(a_logit_when_blinded_model_says_a) + exp(b_logit_when_blinded_model_says_a)),
      pA_given_B = exp(a_logit_when_blinded_model_says_b) /
        (exp(a_logit_when_blinded_model_says_b) + exp(b_logit_when_blinded_model_says_b)),
      marginal_pA = blinded_pA * pA_given_A + (1 - blinded_pA) * pA_given_B,
      marginal_response = log(marginal_pA) - log(1 - marginal_pA)
    )
}

# Load all available sycophancy data
avail_syc <- .available_nicknames("../sycophancy/results", "sycophancy_processed")
combined_sycophancy <- bind_rows(lapply(avail_syc, process_sycophancy_one))

# Sanity-check row counts (60 scenarios * 2 user versions * 4 prompts = 480 per model)
sycophancy_row_counts <- combined_sycophancy %>%
  dplyr::count(model, name = "n_rows")

# Backwards-compat aliases for the original two paper models
Qwen.sycophancy_df <- load_sycophancy_raw("Qwen")
GPT.sycophancy_df <- load_sycophancy_raw("GPT")
Qwen.sycophancy_processed <- combined_sycophancy %>% filter(model == "Qwen")
GPT.sycophancy_processed <- combined_sycophancy %>% filter(model == "GPT")

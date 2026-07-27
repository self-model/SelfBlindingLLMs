# Shared helper functions for demographic bias analysis
# Replaces: loadAndFormatBiasData.R, loadAndFormatBiasToolRespData.R

# Single source of truth: model registry CSV
# Columns: nickname, full_name, family, size_b, in_paper
models_df <- read.csv("model_registry.csv", stringsAsFactors = FALSE)
models_df$size_b <- suppressWarnings(as.numeric(models_df$size_b))
models_df$in_paper <- as.logical(models_df$in_paper)

# Backwards-compat lookup: nickname -> full_name (used in figure paths, etc.)
models <- setNames(models_df$full_name, models_df$nickname)

# Helper: which nicknames have a CSV present in a given results folder
.available_nicknames <- function(results_dir, file_prefix) {
  paths <- file.path(results_dir, sprintf("%s_%s.csv", file_prefix, models_df$full_name))
  models_df$nickname[file.exists(paths)]
}

# Load and prepare demographic bias data from merged CSVs
# nicknames: NULL (= all available), or a character vector of nicknames
load_bias_data <- function(nicknames = NULL) {
  if (is.null(nicknames)) {
    nicknames <- .available_nicknames("../demographic_bias/results",
                                      "demographic_bias_processed")
  }

  load_one <- function(nickname) {
    full_name <- models[[nickname]]
    path <- sprintf("../demographic_bias/results/demographic_bias_processed_%s.csv", full_name)
    if (!file.exists(path)) return(NULL)
    read.csv(path) %>%
      mutate(
        model = nickname,
        model_full = full_name,
        vignette = decision_question_id,
        prompt = prompt_format,
        response = yes_logit - no_logit,
        pyes = exp(response) / (1 + exp(response))
      ) %>%
      # Drop rows with non-finite logits (e.g. Gemini error rows where logits
      # were written as -Inf). Keeps downstream summaries from propagating NaN.
      filter(is.finite(response))
  }

  bind_rows(lapply(nicknames, load_one))
}

# Get responses joined with baseline (removed) responses
with_baseline <- function(df) {
  removed <- df %>%
    filter(prompt == "removed") %>%
    rename(removed_response = response) %>%
    dplyr::select(model, vignette, race, gender, removed_response)

  df %>%
    filter(prompt != "removed") %>%
    left_join(removed, by = c("model", "vignette", "race", "gender")) %>%
    mutate(
      gender = factor(gender, levels = c("male", "female")),
      race = factor(race, levels = c("white", "Black", "Hispanic", "Asian"))
    )
}

# Calculate mean absolute difference from true blindness
summarize_bias <- function(df) {
  df %>%
    mutate(abs_diff = abs(response - removed_response)) %>%
    filter(is.finite(abs_diff)) %>%   # drop NA/Inf (e.g. Gemini error rows)
    group_by(model, prompt) %>%
    summarise(
      mean_abs_diff = mean(abs_diff),
      se_abs_diff = sd(abs_diff) / sqrt(n()),
      .groups = "drop"
    ) %>%
    arrange(-mean_abs_diff) %>%
    mutate(prompt = factor(prompt, levels = unique(prompt)))
}

# Load tool use response data (for conditional logits after tool call)
# Returns wide format with conditional logit columns
load_tool_response_data <- function(nicknames = NULL) {
  if (is.null(nicknames)) {
    nicknames <- .available_nicknames("../demographic_bias/results",
                                      "demographic_bias_processed")
  }

  load_one <- function(nickname) {
    full_name <- models[[nickname]]
    path <- sprintf("../demographic_bias/results/demographic_bias_processed_%s.csv", full_name)
    if (!file.exists(path)) return(NULL)
    read.csv(path) %>%
      mutate(model = nickname,
             model_full = full_name,
             vignette = decision_question_id,
             prompt = prompt_format) %>%
      # Drop rows where any of the four conditional logits is non-finite
      # (Gemini error rows write -Inf). Without this, marginalization NaN-floods.
      filter(is.finite(yes_logit), is.finite(no_logit),
             is.finite(yes_logit_when_tool_says_yes), is.finite(no_logit_when_tool_says_yes),
             is.finite(yes_logit_when_tool_says_no),  is.finite(no_logit_when_tool_says_no))
  }

  bind_rows(lapply(nicknames, load_one))
}

# Reshape conditional logits from wide to long format
# Converts: yes_logit_when_tool_says_yes/no, no_logit_when_tool_says_yes/no
# To: model_response (Yes/No), yes_logit, no_logit
pivot_conditional_logits <- function(df) {
  # Create long format for Yes and No responses
  # Note: This overwrites the original yes_logit/no_logit with the tool-conditioned values
  yes_df <- df %>%
    dplyr::select(-yes_logit, -no_logit,
                  -yes_logit_when_tool_says_no, -no_logit_when_tool_says_no) %>%
    rename(yes_logit = yes_logit_when_tool_says_yes,
           no_logit = no_logit_when_tool_says_yes) %>%
    mutate(model_response = "Yes",
           tool_desc = "run counterfactual simulation")

  no_df <- df %>%
    dplyr::select(-yes_logit, -no_logit,
                  -yes_logit_when_tool_says_yes, -no_logit_when_tool_says_yes) %>%
    rename(yes_logit = yes_logit_when_tool_says_no,
           no_logit = no_logit_when_tool_says_no) %>%
    mutate(model_response = "No",
           tool_desc = "run counterfactual simulation")

  bind_rows(yes_df, no_df)
}

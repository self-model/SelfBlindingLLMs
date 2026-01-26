# Shared helper functions for demographic bias analysis
# Replaces: loadAndFormatBiasData.R, loadAndFormatBiasToolRespData.R

# Single source of truth: nickname -> full model name (as used in filenames)
models <- c(
  "Qwen" = "Qwen2.5-7B-Instruct",
  "GPT" = "GPT-4.1"
)

# Load and prepare demographic bias data from merged CSVs
load_bias_data <- function(model = c("qwen", "gpt", "both")) {
  model <- match.arg(model)

  load_one <- function(nickname) {
    full_name <- models[[nickname]]
    path <- sprintf("../demographic_bias/results/demographic_bias_processed_%s.csv", full_name)
    read.csv(path) %>%
      mutate(
        model = nickname,
        model_full = full_name,
        vignette = decision_question_id,
        prompt = prompt_format,
        response = yes_logit - no_logit,
        pyes = exp(response) / (1 + exp(response))
      )
  }

  qwen <- load_one("Qwen")
  gpt <- load_one("GPT")

  switch(model,
         qwen = qwen,
         gpt = gpt,
         both = bind_rows(qwen, gpt)
  )
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
    group_by(model, prompt) %>%
    summarise(
      mean_abs_diff = mean(abs_diff),
      se_abs_diff = sd(abs_diff) / sqrt(n()),
      .groups = "drop"
    ) %>%
    arrange(-mean_abs_diff) %>%
    mutate(prompt = factor(prompt, levels = prompt))
}

# Load tool use response data (for conditional logits after tool call)
# Returns wide format with conditional logit columns
load_tool_response_data <- function(model = c("qwen", "gpt", "both")) {
  model <- match.arg(model)

  load_one <- function(nickname) {
    full_name <- models[[nickname]]
    path <- sprintf("../demographic_bias/results/demographic_bias_processed_%s.csv", full_name)
    read.csv(path) %>%
      mutate(model = nickname,
             model_full = full_name,
             vignette = decision_question_id,
             prompt = prompt_format)
  }

  qwen <- load_one("Qwen")
  gpt <- load_one("GPT")

  switch(model,
         qwen = qwen,
         gpt = gpt,
         both = bind_rows(qwen, gpt)
  )
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

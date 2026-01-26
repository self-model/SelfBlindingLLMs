# Parse tool call generation data for demographic bias analysis
# Extracts tool call text and checks for race/gender/pronoun inclusion
# Replaces: loadAndFormatBiasToolData.R

# Function to extract tool call text from generations (Qwen format)
extract_tool_call <- function(generations) {
  if (is.null(generations) || length(generations) == 0) {
    return(NA_character_)
  }

  # Get the first generation
  gen_text <- generations[[1]]

  # Check if there's a tool call
  if (grepl("<tool_call>", gen_text, fixed = TRUE)) {
    # Extract the JSON between <tool_call> and </tool_call>
    tool_call_match <- regmatches(gen_text, regexpr("<tool_call>.*?</tool_call>", gen_text))
    if (length(tool_call_match) > 0) {
      # Remove the tags
      tool_call_json <- gsub("</?tool_call>", "", tool_call_match)
      return(tool_call_json)
    }
  }

  return(NA_character_)
}

# Load Qwen tool use generations from JSONL and parse tool calls
# Returns NULL if generation file is not found
load_qwen_tool_generations <- function() {
  qwen_gen_files <- list.files("../demographic_bias/results",
                               pattern = "_tool_use_generations.*\\.jsonl$",
                               full.names = TRUE)

  # Return NULL if no generation file found
  if (length(qwen_gen_files) == 0 || is.na(qwen_gen_files[1])) {
    return(NULL)
  }

  qwen_gen_file <- qwen_gen_files[1]

  stream_in(file(qwen_gen_file), verbose = FALSE) %>%
    filter(!(decision_question_id %in% c(54, 77))) %>%
    mutate(model = 'Qwen') %>%
    select(model, filled_template, decision_question_id, race, gender,
           default, dont_discriminate, ignore, if_you_didnt_know,
           remove_in_context, removed) %>%
    rename(vignette = decision_question_id) %>%
    pivot_longer(
      cols = c(default, dont_discriminate, ignore, if_you_didnt_know,
               remove_in_context, removed),
      names_to = "prompt",
      values_to = "tool_data"
    ) %>%
    unnest_wider(tool_data) %>%
    mutate(
      tool_call_text = map_chr(generations___run_counterfactual_simulation, extract_tool_call),
      has_tool_call = !is.na(tool_call_text),
      tool_call_prompt = map_chr(tool_call_text, function(x) {
        if (is.na(x)) return(NA_character_)
        tryCatch({
          parsed <- fromJSON(x)
          if (!is.null(parsed$arguments) && !is.null(parsed$arguments$prompt)) {
            return(as.character(parsed$arguments$prompt))
          } else {
            return(NA_character_)
          }
        }, error = function(e) {
          return(NA_character_)
        })
      }),
      # Check if race/gender/pronouns included in tool call prompt
      includes_race = if_else(
        has_tool_call & !is.na(tool_call_prompt),
        str_detect(tool_call_prompt, regex(race, ignore_case = TRUE)),
        NA
      ),
      includes_gender = if_else(
        has_tool_call & !is.na(tool_call_prompt),
        str_detect(tool_call_prompt, regex(gender, ignore_case = TRUE)),
        NA
      ),
      includes_gendered_pronouns = if_else(
        has_tool_call & !is.na(tool_call_prompt),
        str_detect(tool_call_prompt, regex("\\b(he|she|his|her|him)\\b", ignore_case = TRUE)),
        NA
      )
    )
}

# Load tool use data with CSV probabilities and generation parsing
load_tool_use_data <- function(model = c("qwen", "gpt", "both")) {
  model <- match.arg(model)

  # Load Qwen: CSV with tool_prob + parsed generations
  # Uses models from helpers.R (must be sourced first)
  qwen_prob <- read.csv(sprintf("../demographic_bias/results/demographic_bias_processed_%s.csv",
                                 models[["Qwen"]])) %>%
    mutate(model = 'Qwen',
           model_full = models[["Qwen"]],
           vignette = decision_question_id,
           prompt = prompt_format)

  qwen_gen <- load_qwen_tool_generations()

  # If generation data exists, join it; otherwise just use prob data
  if (!is.null(qwen_gen)) {
    qwen_combined <- qwen_prob %>%
      left_join(
        qwen_gen %>%
          select(vignette, race, gender, prompt, tool_call_text, has_tool_call,
                 tool_call_prompt, includes_gender, includes_race, includes_gendered_pronouns),
        by = c("vignette", "race", "gender", "prompt")
      )
  } else {
    # No generation data available - add placeholder columns
    qwen_combined <- qwen_prob %>%
      mutate(
        tool_call_text = NA_character_,
        has_tool_call = NA,
        tool_call_prompt = NA_character_,
        includes_gender = NA,
        includes_race = NA,
        includes_gendered_pronouns = NA
      )
  }

  # Load GPT: CSV with tool_prob (no generation parsing available in merged format)
  # Note: GPT tool_prob is aggregated from multiple runs, so we use it as-is
  # Add tool_call as alias for compatibility with scripts that expect boolean
  gpt_combined <- read.csv(sprintf("../demographic_bias/results/demographic_bias_processed_%s.csv",
                                    models[["GPT"]])) %>%
    mutate(model = 'GPT',
           model_full = models[["GPT"]],
           vignette = decision_question_id,
           prompt = prompt_format,
           tool_call = tool_prob)  # tool_prob is already proportion of tool calls

  switch(model,
         qwen = qwen_combined,
         gpt = gpt_combined,
         both = list(qwen = qwen_combined, gpt = gpt_combined)
  )
}

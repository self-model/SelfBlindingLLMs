# Function to extract tool call text from generations
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

# Find Qwen tool use generations file using glob pattern
qwen_gen_file <- list.files("../demographic_bias/results", pattern = "_tool_use_generations.*\\.jsonl$", full.names = TRUE)[1]
Qwen.Tamkin.tool_use_gen <- stream_in(file(qwen_gen_file), verbose = FALSE) %>%
  mutate(model = 'Qwen2.5-7B') %>%
  select(model, filled_template, decision_question_id, race, gender, 
         default, dont_discriminate, ignore, if_you_didnt_know, 
         remove_in_context, removed, redacted) %>%
  rename(vignette = decision_question_id) %>%
  pivot_longer(
    cols = c(default, dont_discriminate, ignore, if_you_didnt_know, 
             remove_in_context, removed, redacted),
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
  ) %>%
  filter(!(vignette %in% c(54, 77)))



# Find Qwen tool use probs file using glob pattern
qwen_prob_file <- list.files("../demographic_bias/results", pattern = "_tool_use_probs.*\\.jsonl$", full.names = TRUE)[1]
Qwen.Tamkin.tool_use_prob <- stream_in(file(qwen_prob_file), verbose = FALSE) %>%
  mutate(model='Qwen2.5-7B')%>%
  # Select the columns we need
  select(model, filled_template, decision_question_id, race, gender, 
         default, dont_discriminate, ignore, if_you_didnt_know, 
         remove_in_context, removed, redact_in_context, redacted) %>%
  rename(vignette = decision_question_id) %>%
  # Pivot prompt columns to long format
  pivot_longer(
    cols = c(default, dont_discriminate, ignore, if_you_didnt_know, 
             remove_in_context, removed, redact_in_context, redacted),
    names_to = "prompt",
    values_to = "tool_data"
  ) %>%
  # Unnest the tool data (each is already a data frame)
  unnest_wider(tool_data) %>%
  # Pivot tool probabilities to long format
  pivot_longer(
    cols = starts_with("tool_prob_with_desc___"),
    names_to = "tool_desc",
    names_prefix = "tool_prob_with_desc___",
    values_to = "tool_prob"
  ) %>%
  # Clean up tool descriptions
  mutate(
    tool_desc = str_replace_all(tool_desc, "_", " ")
  ) %>%
  filter(!(vignette %in% c(54,77))) # inverted


# Merge them
Qwen.Tamkin.tool_use_combined <- Qwen.Tamkin.tool_use_prob %>%
  left_join(
    Qwen.Tamkin.tool_use_gen %>% 
      select(vignette, race, gender, prompt, tool_call_text, has_tool_call, 
             tool_call_prompt, includes_gender, includes_race, includes_gendered_pronouns),
    by = c("vignette", "race", "gender", "prompt")
  )

# Check the result
colnames(Qwen.Tamkin.tool_use_combined)

# GPT

GPT.Tamkin.tool_use_combined <- read.csv('../demographic_bias/results/GPT tool calls/tool_use.csv')

# 
# # Function to extract tool call text
# extract_tool_call_text <- function(completion_json_str) {
#   if (is.na(completion_json_str) || completion_json_str == "" || is.null(completion_json_str)) {
#     return(NA_character_)
#   }
#   
#   tryCatch({
#     # Parse the JSON string
#     completion <- fromJSON(completion_json_str, simplifyVector = FALSE)
#     
#     # Navigate to tool_calls
#     if (!is.null(completion$choices) && length(completion$choices) > 0) {
#       message <- completion$choices[[1]]$message
#       
#       if (!is.null(message$tool_calls) && length(message$tool_calls) > 0) {
#         # Extract the arguments from the first tool call
#         tool_call <- message$tool_calls[[1]]
#         # Use bracket notation instead of $ for 'function'
#         if (!is.null(tool_call[["function"]]$arguments)) {
#           return(as.character(tool_call[["function"]]$arguments))
#         }
#       }
#     }
#     
#     # No tool calls found
#     return(NA_character_)
#     
#   }, error = function(e) {
#     return(NA_character_)
#   })
# }
# 
# # Function to check if there was a tool call - DEBUGGED VERSION
# check_tool_call <- function(completion_json_str) {
#   if (is.na(completion_json_str) || completion_json_str == "" || is.null(completion_json_str)) {
#     return(NA)
#   }
#   tryCatch({
#     # Parse the JSON string
#     completion <- fromJSON(completion_json_str, simplifyVector = FALSE)
#     # Navigate to tool_calls
#     if (!is.null(completion$choices) && length(completion$choices) > 0) {
#       message <- completion$choices[[1]]$message
#       if (!is.null(message$tool_calls)) {
#         # Found tool calls!
#         return(TRUE)
#       }
#     }
#     # No tool calls found
#     return(FALSE)
#   }, error = function(e) {
#     cat(sprintf("Error parsing JSON: %s\n", e$message))
#     return(NA)
#   })
# }
# 
# # Get all run files
# gpt_run_files <- list.files(
#   path = "../demographic_bias/results/GPT tool calls",
#   pattern = "run\\d+\\.jsonl$",
#   full.names = TRUE
# )
# 
# # Sort files to ensure consistent ordering
# gpt_run_files <- sort(gpt_run_files)
# cat(sprintf("Found %d run files\n", length(gpt_run_files)))
# 
# # Function to process a single file with sequential run number
# process_gpt_run_file <- function(filepath, run_number) {
#   cat(sprintf("\nProcessing [%d/%d]: %s\n", run_number, length(gpt_run_files), basename(filepath)))
#   
#   # Use sequential number as run ID
#   run_id <- sprintf("run%03d", run_number)
#   
#   # Read the JSONL file
#   data <- stream_in(file(filepath), verbose = FALSE)
#   cat(sprintf("  Rows: %d\n", nrow(data)))
#   
#   # Get all column names that match the pattern {prompt}__{tool}__completion_json
#   completion_cols <- names(data)[str_detect(names(data), "__.*__completion_json$")]
#   
#   # Extract prompt and tool from each column name
#   result_list <- list()
#   
#   for (col in completion_cols) {
#     parts <- str_split(col, "__")[[1]]
#     prompt <- parts[1]
#     tool_desc <- parts[2]
#     
#     # SKIP if not run_counterfactual_simulation
#     if (tool_desc != "run_counterfactual_simulation") {
#       next
#     }
#     
#     temp_df <- data %>%
#       select(decision_question_id, gender, race, scenario_id, filled_template, all_of(col)) %>%
#       rename(completion_json = all_of(col)) %>%
#       mutate(
#         run = run_id,
#         model = "GPT4.1",
#         prompt = prompt,
#         tool_desc = tool_desc,
#         tool_call = map_lgl(completion_json, check_tool_call),
#         tool_call_text = map_chr(completion_json, extract_tool_call_text)  # NEW COLUMN
#       ) %>%
#       select(run, model, decision_question_id, gender, race, scenario_id,
#              filled_template, prompt, tool_desc, tool_call, tool_call_text)  # Include new column
#     
#     result_list[[col]] <- temp_df
#   }
#   
#   result <- bind_rows(result_list)
#   
#   cat(sprintf("  Output rows: %d\n", nrow(result)))
#   cat(sprintf("  Tool calls detected: %d (%.1f%%)\n",
#               sum(result$tool_call, na.rm = TRUE),
#               100 * mean(result$tool_call, na.rm = TRUE)))
#   
#   return(result)
# }
# 
# # Process all files with sequential numbering
# cat("\n=== Processing all run files ===\n")
# gpt_tool_calls_df <- map2_df(
#   gpt_run_files,
#   seq_along(gpt_run_files),
#   process_gpt_run_file
# ) %>%
#   mutate(
#     tool_call_prompt = map_chr(tool_call_text, function(x) {
#       if (is.na(x)) return(NA_character_)
#       tryCatch({
#         parsed <- fromJSON(x)
#         return(parsed$prompt)
#       }, error = function(e) NA_character_)
#     })
#   ) %>%
#   mutate(
#     tool_call_prompt = map_chr(tool_call_text, function(x) {
#       if (is.na(x)) return(NA_character_)
#       tryCatch({
#         parsed <- fromJSON(x)
#         return(parsed$prompt)
#       }, error = function(e) NA_character_)
#     }),
#     # Check if race/gender/pronouns included in tool call prompt
#     includes_race = if_else(
#       tool_call & !is.na(tool_call_prompt),
#       str_detect(tool_call_prompt, regex(race, ignore_case = TRUE)),
#       NA
#     ),
#     includes_gender = if_else(
#       tool_call & !is.na(tool_call_prompt),
#       str_detect(tool_call_prompt, regex(gender, ignore_case = TRUE)),
#       NA
#     ),
#     includes_gendered_pronouns = if_else(
#       tool_call & !is.na(tool_call_prompt),
#       str_detect(tool_call_prompt, regex("\\b(he|she|his|her|him)\\b", ignore_case = TRUE)),
#       NA
#     )
#   )
# 
# gpt_tool_calls_df %>% write.csv('../demographic_bias/results/GPT tool calls/tool_use.csv')
# # Load Qwen data (single file)
# Qwen_resp_raw <- stream_in(
#   file("../results/tool use/Tamkin/responses/20260108_193428_explicit_prompts_Qwen2.5-7B-Instruct_tool_use_response_logprobs.jsonl"), 
#   verbose = FALSE
# )
# 
# # Get all GPT run files
# gpt_run_files <- list.files(
#   path = "../results/tool use/Tamkin/responses/GPT",
#   pattern = "\\.jsonl$",
#   full.names = TRUE
# )
# 
# # Sort files to ensure consistent ordering
# gpt_run_files <- sort(gpt_run_files)
# cat(sprintf("Found %d GPT run files\n", length(gpt_run_files)))
# 
# # Load all GPT files with run identifiers
# GPT_resp_raw_list <- map2(
#   gpt_run_files,
#   seq_along(gpt_run_files),
#   function(filepath, run_num) {
#     cat(sprintf("Loading run %d: %s\n", run_num, basename(filepath)))
#     data <- stream_in(file(filepath), verbose = FALSE)
#     data$run <- sprintf("run%03d", run_num)  # Add run identifier
#     return(data)
#   }
# )
# 
# # Combine all GPT runs
# GPT_resp_raw_combined <- bind_rows(GPT_resp_raw_list)
# 
# # ------------------------------------------------------------------------------
# # Extract and reshape logits data from nested structure
# # ------------------------------------------------------------------------------
# 
# process_model_responses_fast <- function(resp_raw, model_name, prompts, tools) {
#   
#   cat(sprintf("\n=== Processing %s data ===\n", model_name))
#   cat(sprintf("Total rows: %d\n", nrow(resp_raw)))
#   
#   # Identify base columns that exist
#   base_cols <- c("filled_template", "decision_question_id", "race", "gender")
#   if ("run" %in% names(resp_raw)) {
#     base_cols <- c(base_cols, "run")
#     cat(sprintf("Found %d unique runs\n", n_distinct(resp_raw$run)))
#   }
#   
#   cat("Step 1/6: Selecting and renaming columns...\n")
#   result <- resp_raw %>%
#     select(all_of(base_cols), any_of(prompts)) %>%
#     rename(vignette = decision_question_id)
#   
#   cat(sprintf("Step 2/6: Pivoting prompts to long format (%d prompts)...\n", length(prompts)))
#   result <- result %>%
#     pivot_longer(
#       cols = any_of(prompts),
#       names_to = "prompt",
#       values_to = "prompt_data"
#     )
#   cat(sprintf("  Rows after pivot: %d\n", nrow(result)))
#   
#   cat("Step 3/6: Unnesting prompt data...\n")
#   result <- result %>%
#     unnest_wider(prompt_data)
#   
#   cat(sprintf("Step 4/6: Pivoting tools to long format (%d tools)...\n", length(tools)))
#   result <- result %>%
#     pivot_longer(
#       cols = any_of(tools),
#       names_to = "tool_desc",
#       values_to = "tool_data"
#     )
#   cat(sprintf("  Rows after pivot: %d\n", nrow(result)))
#   
#   cat("Step 5/6: Unnesting tool data...\n")
#   result <- result %>%
#     unnest_wider(tool_data)
#   
#   cat("Step 6/6: Pivoting Yes/No responses and unnesting logits...\n")
#   result <- result %>%
#     pivot_longer(
#       cols = c(`Yes.`, `No.`),
#       names_to = "model_response",
#       values_to = "logits_data"
#     ) %>%
#     mutate(model_response = str_remove(model_response, "\\.$")) %>%
#     unnest_wider(logits_data) %>%
#     mutate(model = model_name)
#   
#   cat(sprintf("✓ Complete! Final rows: %d\n", nrow(result)))
#   
#   return(result)
# }
# 
# prompts <- c("default", "dont_discriminate", "ignore", "if_you_didnt_know", "removed", "redacted")
# tools <- c("call_yourself", "run_counterfactual_simulation", "controlled_reprompt")
# 
# GPT_resp_df <- process_model_responses_fast(GPT_resp_raw_combined, "GPT", prompts, tools) %>%
#   mutate(tool_desc = str_replace_all(tool_desc, "_", " "))
# 
# Qwen_resp_df <- process_model_responses_fast(Qwen_resp_raw, "Qwen", prompts, tools) %>%
#   mutate(tool_desc = str_replace_all(tool_desc, "_", " "))
# 
# 
# # Process both models
# Qwen_resp_df <- process_model_responses(Qwen_resp_raw, "Qwen", prompts, tools) %>%
#   mutate(tool_desc = str_replace_all(tool_desc, "_", " "))
# 
# GPT_resp_df <- process_model_responses(GPT_resp_raw, "GPT", prompts, tools) %>%
#   mutate(tool_desc = str_replace_all(tool_desc, "_", " "))
# 
# # Combine both models
# combined_resp_df <- bind_rows(Qwen_resp_df, GPT_resp_df %>%
#                                 group_by(filled_template,vignette,race,gender,tool_desc,model, prompt,model_response) %>%
#                                 summarise(no_logit=mean(no_logit),
#                                           yes_logit=mean(yes_logit)))
# 
# GPT_resp_df %>%
#   write.csv("../results/tool use/Tamkin/responses/GPT.csv")
# 
# Qwen_resp_df %>%
#   write.csv("../results/tool use/Tamkin/responses/Qwen.csv")
# 
# combined_resp_df %>%
#   write.csv("../results/tool use/Tamkin/responses/combined.csv")

GPT_resp_df <- read.csv("../results/tool use/Tamkin/responses/GPT.csv")

Qwen_resp_df <- read.csv("../results/tool use/Tamkin/responses/Qwen.csv")

combined_resp_df <- read.csv("../results/tool use/Tamkin/responses/combined.csv")

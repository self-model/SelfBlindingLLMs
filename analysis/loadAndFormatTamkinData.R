# Find Qwen yes/no file using glob pattern
qwen_yn_file <- list.files("../demographic_bias/results", pattern = "_yn_.*Qwen.*\\.jsonl$", full.names = TRUE)[1]
Qwen.Tamkin.YN.df <- stream_in(file(qwen_yn_file), verbose=FALSE) %>% 
  mutate(model='Qwen') %>%
  dplyr::select(-matches("relative_probs")) %>%
  pivot_longer(
    cols = matches("_prompt_"),
    names_to = c("prompt", "measure"),
    names_pattern = "(.*)_prompt_(.*)",
    values_to = "value_raw"
  ) %>%
  pivot_wider(
    names_from = measure,   # "yes_logits", "no_logits"
    values_from = value_raw
  ) %>%
  mutate(response = yes_logits-no_logits,
         pyes=exp(response)/(1+exp(response))) %>%
  rename(vignette = decision_question_id) %>%
  filter(!(vignette %in% c(54,77))) # inverted

Qwen.Tamkin.removed_responses <- Qwen.Tamkin.YN.df %>% 
  filter(prompt=='removed') %>%
  rename(removed_response=response) %>%
  dplyr::select(model, vignette,race,gender,removed_response)

Qwen.Tamkin.other_responses <-Qwen.Tamkin.YN.df %>% 
  filter(prompt!='removed') %>%
  dplyr::select(model, vignette,race,gender,prompt, response)

Qwen.Tamkin.response_by_prompt <- Qwen.Tamkin.other_responses %>%
  merge(Qwen.Tamkin.removed_responses) %>%
  mutate(
    gender = factor(gender, levels = c("male", "female")),  # male as baseline
    race = factor(race, levels = c("white", "Black", "Hispanic", "Asian"))       # white as baseline
  )

# Calculate mean absolute difference and standard errors for each prompt
Qwen.Tamkin.mean_abs_diff <- Qwen.Tamkin.response_by_prompt %>%
  mutate(abs_diff = abs(response - removed_response)) %>%
  group_by(model, prompt) %>%
  summarise(
    mean_abs_diff = mean(abs_diff),
    se_abs_diff = sd(abs_diff) / sqrt(n()),
    .groups = "drop"
  ) %>%
  # Order by mean_abs_diff (best to worst)
  arrange(-mean_abs_diff) %>%
  mutate(prompt = factor(prompt, levels = prompt))

gpt_path <- "../demographic_bias/results/GPT-Tamkin"

GPT.Tamkin.YN.wide.df <- list.files(gpt_path, pattern = "\\.jsonl$", full.names = TRUE) %>%
  # Convert to tibble first so we keep filename
  tibble(filename = .) %>%
  mutate(
    base  = tools::file_path_sans_ext(basename(filename)),  # drop dir + extension
    model = str_extract(base, "^[^_]+"),                   # before first _
    run   = str_extract(base, "[^_]+$")                    # after last _
  ) %>%
  dplyr::select(-base) %>%
  mutate(data = map(filename, ~ stream_in(file(.), verbose = FALSE))) %>%
  unnest(data) %>%
  mutate(model = 'GPT') 

GPT.Tamkin.YN.df <- GPT.Tamkin.YN.wide.df %>%
  dplyr::select(-matches("relative_probs")) %>%
  pivot_longer(
    cols = matches("_prompt_"),
    names_to = c("prompt", "measure"),
    names_pattern = "(.*)_prompt_(.*)",
    values_to = "value_raw"
  ) %>%
  pivot_wider(
    names_from = measure,   
    values_from = value_raw
  )%>%
  mutate(pyes=exp(yes_logits)/(exp(yes_logits)+exp(no_logits)),
         response=yes_logits-no_logits) %>%
  group_by(model,across(filled_template:decision_question_id),prompt) %>%
  summarise(
    se_pyes=se(pyes),
    pyes=mean(pyes),
    se_response=se(response),
    response=mean(response)) %>%
  rename(vignette = decision_question_id) %>%
  filter(!(vignette %in% c(54,77))) # inverted


GPT.Tamkin.removed_responses <- GPT.Tamkin.YN.df %>% 
  filter(prompt=='removed') %>%
  rename(removed_response=response) %>%
  dplyr::select(model, vignette,race,gender,removed_response)

GPT.Tamkin.other_responses <-GPT.Tamkin.YN.df %>% 
  filter(prompt!='removed') %>%
  dplyr::select(model, vignette,race,gender,prompt, response)

GPT.Tamkin.response_by_prompt <- GPT.Tamkin.other_responses %>%
  merge(GPT.Tamkin.removed_responses) %>%
  mutate(
    gender = factor(gender, levels = c("male", "female")),  # male as baseline
    race = factor(race, levels = c("white", "Black", "Hispanic", "Asian"))       # white as baseline
  )

# Calculate mean absolute difference and standard errors for each prompt
GPT.Tamkin.mean_abs_diff <- GPT.Tamkin.response_by_prompt %>%
  mutate(abs_diff = abs(response - removed_response)) %>%
  group_by(model, prompt) %>%
  summarise(
    mean_abs_diff = mean(abs_diff),
    se_abs_diff = sd(abs_diff) / sqrt(n()),
    .groups = "drop"
  ) %>%
  # Order by mean_abs_diff (best to worst)
  arrange(-mean_abs_diff) %>%
  mutate(prompt = factor(prompt, levels = prompt))

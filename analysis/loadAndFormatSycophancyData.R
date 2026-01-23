Qwen.sycophancy_df <- read.csv('../sycophancy/results/sycophancy_processed_qwen2.5-7b-instruct.csv') %>%
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

GPT.sycophancy_df <- read.csv('../sycophancy/results/sycophancy_processed_gpt-4.1.csv') %>%
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

# Process Qwen data
Qwen.sycophancy_processed <- Qwen.sycophancy_df %>%
  group_by(scenario_id, my_version, prompt) %>%
  summarise(
    # Average logits over presentation orders (my_first)
    a_logit = mean(a_logit),
    b_logit = mean(b_logit),
    blinded_model_a_logit = mean(blinded_model_a_logit),
    blinded_model_b_logit = mean(blinded_model_b_logit),
    # Self-call logits
    a_logit_when_blinded_model_says_a = mean(a_logit_when_blinded_model_says_a),
    b_logit_when_blinded_model_says_a = mean(b_logit_when_blinded_model_says_a),
    a_logit_when_blinded_model_says_b = mean(a_logit_when_blinded_model_says_b),
    b_logit_when_blinded_model_says_b = mean(b_logit_when_blinded_model_says_b),
    .groups = "drop"
  ) %>%
  mutate(
    model = "Qwen",
    # Response = preference for A over B
    response = a_logit - b_logit,
    # Removed response = blind preference for A over B
    removed_response = blinded_model_a_logit - blinded_model_b_logit,
    # Marginal response calculations
    blinded_pA = exp(blinded_model_a_logit) / (exp(blinded_model_a_logit) + exp(blinded_model_b_logit)),
    pA_given_A = exp(a_logit_when_blinded_model_says_a) / 
      (exp(a_logit_when_blinded_model_says_a) + exp(b_logit_when_blinded_model_says_a)),
    pA_given_B = exp(a_logit_when_blinded_model_says_b) / 
      (exp(a_logit_when_blinded_model_says_b) + exp(b_logit_when_blinded_model_says_b)),
    marginal_pA = blinded_pA * pA_given_A + (1 - blinded_pA) * pA_given_B,
    marginal_response = log(marginal_pA) - log(1 - marginal_pA)
  )

# Process GPT data
GPT.sycophancy_processed <- GPT.sycophancy_df %>%
  group_by(scenario_id, my_version, prompt) %>%
  summarise(
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
    model = "GPT",
    response = a_logit - b_logit,
    removed_response = blinded_model_a_logit - blinded_model_b_logit,
    blinded_pA = exp(blinded_model_a_logit) / (exp(blinded_model_a_logit) + exp(blinded_model_b_logit)),
    pA_given_A = exp(a_logit_when_blinded_model_says_a) / 
      (exp(a_logit_when_blinded_model_says_a) + exp(b_logit_when_blinded_model_says_a)),
    pA_given_B = exp(a_logit_when_blinded_model_says_b) / 
      (exp(a_logit_when_blinded_model_says_b) + exp(b_logit_when_blinded_model_says_b)),
    marginal_pA = blinded_pA * pA_given_A + (1 - blinded_pA) * pA_given_B,
    marginal_response = log(marginal_pA) - log(1 - marginal_pA)
  )

# Check dimensions
cat(sprintf("Qwen: %d rows (should be 60 scenarios × 2 user versions × 5 prompts = 600)\n", 
            nrow(Qwen.sycophancy_processed)))
cat(sprintf("GPT: %d rows (should be 60 scenarios × 2 user versions × 5 prompts = 600)\n", 
            nrow(GPT.sycophancy_processed)))

# Combine
combined_sycophancy <- bind_rows(Qwen.sycophancy_processed, GPT.sycophancy_processed)
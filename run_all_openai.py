set -e  # stop on first error
cd /content/public_repo  # adjust if your clone is elsewhere
                                                                                                                                        
# Verify the API key is loaded (script uses python-dotenv → .env)                                                                         
python -c "from dotenv import load_dotenv; import os; load_dotenv(); assert os.getenv('OPENAI_API_KEY'), 'OPENAI_API_KEY not in env or    
.env'; print('✓ OPENAI_API_KEY present')"                                                                                                 
                                                    
for MODEL in gpt-4.1-mini gpt-4.1-nano; do                                                                                                
echo ""                                             
echo "============================================================"                                                                     
echo "MODEL: $MODEL"                                
echo "============================================================"

# Demographic bias (3 tasks)                                                                                                            
python -m demographic_bias.inference.yn_logprobs_openai            --openai_model "$MODEL" --batch
python -m demographic_bias.inference.tool_use_probs_openai         --openai_model "$MODEL"                                              
python -m demographic_bias.inference.tool_result_yn_logprobs_openai --openai_model "$MODEL"
                                                                                                                                        
# Sycophancy (4 tasks)                              
python -m sycophancy.inference.first_person_openai           --openai_model "$MODEL" --batch                                            
python -m sycophancy.inference.third_person_openai           --openai_model "$MODEL" --batch                                            
python -m sycophancy.inference.tool_use_probs_openai         --openai_model "$MODEL"
python -m sycophancy.inference.tool_result_yn_logprobs_openai --openai_model "$MODEL" --batch                                           
                                                                                                                                        
# Build the per-experiment processed CSVs for this model                                                                                
python demographic_bias/build_csv.py --model "$MODEL"                                                                                   
python sycophancy/build_csv.py        --model "$MODEL"                                                                                  
done                                                                                                                                      

echo ""                                                                                                                                   
echo "DONE. Outputs in demographic_bias/results/ and sycophancy/results/"

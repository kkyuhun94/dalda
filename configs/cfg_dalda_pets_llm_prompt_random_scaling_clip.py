EXP_NAME = "dalda-pets-llm-prompt-random-scaling-clip"

model_path = "CompVis/stable-diffusion-v1-4"

dataset = "pets"

logdir = f"{dataset}-baselines/{EXP_NAME}"
synthetic_dir = f"aug/{EXP_NAME}" + "/{dataset}-{seed}-{examples_per_class}"

prompt_mode = "llm" # Options : ["llm", "base"]
llm_prompt_path = "prompt_by_LLM/llm_prompt_pets_10.json" # Only for LLM prompt mode

synthetic_probability = 0.5
guidance_scale = 7.5
num_synthetic = 10
num_trials = 3
examples_per_class = [1, 2, 4, 8, 16]

image_size = 224
classifier_backbone = "clip"
iterations_per_epoch = 200
num_epochs = 50
batch_size = 32
num_workers = 12
device = 'cuda:0' # Default : 'cuda:0'

aug = "dalda"

scale = "all_random" # Options : ["all_random", "adaptive", 0.1, 0.2, ..., 1.0]
num_inference_steps = 30

resume = [None, None, None, None] # Default : [None, None, None, None]
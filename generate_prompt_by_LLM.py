import os
import json
import time

from dotenv import load_dotenv
from openai import AzureOpenAI

from semantic_aug.datasets.flowers102 import Flowers102Dataset
from semantic_aug.datasets.pets import PetsDataset
from semantic_aug.datasets.caltech101 import CalTech101Dataset

load_dotenv()

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_ENDPOINT"), 
  api_key= os.getenv("AZURE_API_KEY"),  
  api_version= os.getenv("AZURE_API_VERSION")
)

GPT_MODEL = os.getenv("AZURE_MODEL")

######################## Configuration ########################

DATASET = "pets"
NUM_CLASS = 37

USER_PROMPT_FILE_NAME = f"llm_prompt_1_user_{DATASET}.txt"
SYSTEM_PROMPT_FILE_NAME = f"llm_prompt_1_system_{DATASET}.txt"
PROMPT_JSON_NAME = "llm_prompt_{DATASET}_{trial}.json"

NUM_PROMPTS = 10

##############################################################

def read_text(file_path: str):
    with open(file_path, "r") as f:
        return f.read()

def write_json(file_path: str, content: dict):
    with open(file_path, "w") as f:
        json.dump(content, f)

def read_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

def get_class_names(dataset: str):
    if 'flowers' in dataset:
        class_names = Flowers102Dataset().class_names
    elif "pets" in dataset:
        class_names = PetsDataset().class_names
    elif "caltech" in dataset:
        class_names = CalTech101Dataset().class_names

    return class_names

def get_gpt_response(user_prompt, system_prompt):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

if __name__ == "__main__":

    prompt_for_llm_dir_path = "prompt_for_LLM"
    prompt_by_llm_dir_path = "prompt_by_LLM"

    os.makedirs(prompt_for_llm_dir_path, exist_ok=True)
    os.makedirs(prompt_by_llm_dir_path, exist_ok=True)

    user_prompt_path = os.path.join(prompt_for_llm_dir_path, USER_PROMPT_FILE_NAME)
    system_prompt_path = os.path.join(prompt_for_llm_dir_path, SYSTEM_PROMPT_FILE_NAME)

    user_prompt = read_text(user_prompt_path)
    system_prompt = read_text(system_prompt_path)

    for i in range(NUM_PROMPTS):
	MAX_RETRIES = 10
        start_time = time.time()
        print(f"Trial {i} started.")

        while MAX_RETRIES > 0:
            try:
                response, input_tokens, output_tokens = get_gpt_response(user_prompt, system_prompt)

                content_dict = json.loads(response)

                generated_prompt_path = os.path.join(prompt_by_llm_dir_path, PROMPT_JSON_NAME.format(DATASET = DATASET, trial = i))
                
                if len(content_dict) == NUM_CLASS:
                    write_json(generated_prompt_path, content_dict)
                    MAX_RETRIES = 0
                else:
                    MAX_RETRIES -= 1
                    print(f"The length of the generated prompt is not {NUM_CLASS}. Retrying. Remaining retries: {MAX_RETRIES}")
                    time.sleep(2)
                    continue
                
            except json.decoder.JSONDecodeError as e:
                MAX_RETRIES -= 1
                print(f"JSON Decode Error occurred. Retrying. Remaining retries: {MAX_RETRIES}")
                time.sleep(2)
            except Exception as e:
                MAX_RETRIES -= 1
                print(f"An error occurred. Retrying. Remaining retries: {MAX_RETRIES}")
                print(e)
                time.sleep(2)
            
        end_time = time.time()
        total_time = time.strftime('%M minutes %S seconds', time.gmtime(end_time - start_time))

        print(f"Trial {i} completed.")
        print(f"Time taken: {total_time}")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")

    ## Combine individual prompts into one json file after generating each prompt
    class_names_prompted = [class_name.strip() for class_name in user_prompt.split("<class>")[-1].replace("'", "").split(",")]

    single_prompt_paths = []

    for i in range(NUM_PROMPTS):
        single_prompt_file_path = f'{prompt_by_llm_dir_path}/{PROMPT_JSON_NAME.format(DATASET = DATASET, trial = i)}'
        
        single_prompt_paths.append(single_prompt_file_path)

        prompt_dict = json.load(open(single_prompt_file_path))
        prompt_dict = {key.strip(): value.encode('utf-8').decode('unicode_escape') for key, value in prompt_dict.items()}
        prompt_dict = dict(zip(class_names_prompted, prompt_dict.values()))

        if i == 0:
            prompt_dict_new = {key: [value] for key, value in prompt_dict.items()}
            continue
        
        for key, value in prompt_dict.items():
            prompt_dict_new[key].append(value)

    class_names = get_class_names(DATASET)
    prompt_dict_new = dict(zip(class_names, prompt_dict_new.values()))

    with open(f'{prompt_by_llm_dir_path}/{PROMPT_JSON_NAME.format(DATASET = DATASET, trial = NUM_PROMPTS)}', 'w', encoding = 'utf-8') as json_file:
        json.dump(prompt_dict_new, json_file, ensure_ascii = False)

    for single_prompt_path in single_prompt_paths:
        os.remove(single_prompt_path)

    print(f"Prompt json file generated successfully. ({NUM_CLASS} classes)")

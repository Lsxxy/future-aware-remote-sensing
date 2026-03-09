import json
import re
import os
import yaml
import numpy as np
from tqdm import tqdm
import torch
import argparse
from PIL import Image
from peft import PeftModel, PeftConfig


from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
# 确保你的 utils.py 文件包含了我们之前讨论的所有评测函数
from utils import extract_characters_regex, calculate_bleu_scores, calculate_iou_accuracy, calculate_vqa_accuracy, calculate_mme_rs_accuracy

PROMPT_LIBRARY_PATH = './prompt_library.yaml' 
try:
    with open(PROMPT_LIBRARY_PATH, 'r', encoding='utf-8') as f:
        PROMPTS = yaml.safe_load(f)
    print("✅ Prompt library loaded successfully.")
except FileNotFoundError:
    print(f"❌ ERROR: Prompt library file not found at {PROMPT_LIBRARY_PATH}. Please create it.")
    PROMPTS = {} 
except Exception as e:
    print(f"❌ ERROR: Failed to load or parse {PROMPT_LIBRARY_PATH}: {e}")
    PROMPTS = {}

def get_language(text: str) -> str:
    if text and re.search(r"[一-龥]", text):
        return "zh"
    return "en"

def load_all_test_data(tasks_to_run, caption_files=None, choices_files=None):
    """
    根据指定的任务列表和直接的文件路径列表加载测试数据。
    """
    tasks = []
    
    # 任务类型映射
    task_type_mapping = {'caption': 'Image caption', 'choices': 'choices'}

    # 如果要运行Caption任务，并且提供了文件路径
    if 'caption' in tasks_to_run and caption_files:
        print("Loading caption tasks from provided files...")
        for file_path in caption_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    print(f"  - Reading {file_path}")
                    for item in json.load(f):
                        tasks.append({
                            'task_id': item['Question id'],
                            'image_path': item['Image'],
                            'task_type': item.get('Task', 'Image caption'), # Safely get task type
                            'prompt_info': {'question': item['Text']},
                            'ground_truth': item['Ground truth']
                        })
            except FileNotFoundError:
                print(f"❌ ERROR: Caption file not found at {file_path}. Please check the path.")
            except Exception as e:
                print(f"❌ ERROR: Failed to read or parse {file_path}: {e}")

    # 如果要运行选择题任务，并且提供了文件路径
    if 'choices' in tasks_to_run and choices_files:
        print("Loading choices tasks from provided files...")
        for file_path in choices_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    print(f"  - Reading {file_path}")
                    for item in json.load(f):
                        tasks.append({
                            'task_id': item['Question id'],
                            'image_path': item['Image'],
                            'task_type': 'choices',
                            'prompt_info': {
                                'question': item['Text'],
                                'choices': item['Answer choices']
                            },
                            'ground_truth': item['Ground truth']
                        })
            except FileNotFoundError:
                print(f"❌ ERROR: Choices file not found at {file_path}. Please check the path.")
            except Exception as e:
                print(f"❌ ERROR: Failed to read or parse {file_path}: {e}")

    return tasks

def build_prompt_for_task(task):
    """
    为单个任务构建prompt，并能根据任务语言（中文/英文）从加载的PROMPTS库中自动选择对应的硬提示。
    """
    if not PROMPTS: 
        return task['prompt_info'].get('question', '')

    lang = 'en' # 默认语言为英文
    
    if task['task_type'] == 'Image caption':
        text_for_lang_detection = task.get('ground_truth', '')
        lang = get_language(text_for_lang_detection)
        
        prompt = PROMPTS[lang]['caption_sys_prompt']
        

        
    elif task['task_type'] == 'choices':
        text_for_lang_detection = task['prompt_info'].get('question', '')
        lang = get_language(text_for_lang_detection)
        
        sys_prompt = PROMPTS[lang]['choices_sys_prompt']
        
        question = task['prompt_info']['question']
        choices_str = "\n".join(task['prompt_info']['choices'])
        
        final_answer_prompt = "最终答案是:" if lang == 'zh' else "The best answer is:"
        prompt = f"{question}\n{choices_str}\n\n{sys_prompt}\n\n{final_answer_prompt}"
        
    else:
        prompt = task['prompt_info'].get('question', '')

    return prompt

def calculate_final_scores(results, tasks_to_run, save_path=None):
    """根据运行的任务计算最终分数，并生成动态报告。"""
    S_final, S1, S2 = 0, 0, 0
    X1, X2 = 0, 0
    bleu_results = {}
    report_data = {"overall_scores": {}, "task_details": {}}

    if 'caption' in tasks_to_run:
        caption_results = [r for r in results if r['task_type'] == 'Image caption']
        if caption_results:
            gt_captions = [r['ground_truth'] for r in caption_results]
            pred_captions = [r['model_output'] for r in caption_results]
            bleu_results = calculate_bleu_scores(gt_captions, pred_captions)
            X1 = bleu_results.get('Avg_BLEU', 0)
            S1 = X1 * 15
            S_final += S1
            report_data["overall_scores"]["S1_Caption"] = round(S1, 4)
            report_data["task_details"]["Caption"] = {"score": round(S1, 4), "metric": "Avg_BLEU", "value": round(X1, 4), "bleu_details": {k: round(v, 4) for k, v in bleu_results.items()}}
        else:
            print("Warning: 'caption' task was specified, but no caption results found for scoring.")

    if 'choices' in tasks_to_run:
        choices_results = [r for r in results if r['task_type'] == 'choices']
        if choices_results:
            gt_choices = [r['ground_truth'] for r in choices_results]
            pred_choices = [r['model_output'] for r in choices_results]
            X2 = calculate_mme_rs_accuracy(gt_choices, pred_choices)
            S2 = X2 * 45
            S_final += S2
            report_data["overall_scores"]["S2_Choices"] = round(S2, 4)
            report_data["task_details"]["Choices"] = {"score": round(S2, 4), "metric": "Accuracy", "value": round(X2, 4)}
        else:
            print("Warning: 'choices' task was specified, but no choices results found for scoring.")

    report_data["overall_scores"]["S_final_combined"] = round(S_final, 4)
    if save_path:
        try:
            os.makedirs(save_path, exist_ok=True)
            output_file_path = os.path.join(save_path, 'results.json')
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)
            print(f"Final scores report saved to {output_file_path}")
        except Exception as e:
            print(f"❌ Error saving report to {save_path}: {e}")

    print("\n" + "="*40 + "\n------ Valid Performance Report ------\n" + "="*40)
    if 'caption' in tasks_to_run and bleu_results:
        print(f"Caption Score (S1): {S1:.2f} / 15\n  - Avg_BLEU (X1 = {X1:.4f}) -> Score: {S1:.2f}")
        print(f"    (Details: BLEU-1: {bleu_results.get('BLEU-1', 0):.3f}, BLEU-2: {bleu_results.get('BLEU-2', 0):.3f}, BLEU-4: {bleu_results.get('BLEU-4', 0):.3f})")
        print("-" * 20)
    if 'choices' in tasks_to_run and X2 > 0:
        print(f"Choices Score (S2): {S2:.2f} / 45\n  - Accuracy (X2 = {X2:.4f}) -> Score: {S2:.2f}")
        print("-" * 20)
    print(f"Final Combined Score (S): {S_final:.2f}")
    print("="*40)
    return S_final

def run_batch_inference(model, tokenizer, tasks, image_base_path, batch_size=8, task_name=""):
    results = []
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        progress_bar_desc = f"Running Inference for {task_name}"
        for i in tqdm(range(0, len(tasks), batch_size), desc=progress_bar_desc):
            batch_tasks = tasks[i:i+batch_size]
            batch_msgs = []
            for task in batch_tasks:
                full_image_path = os.path.join(image_base_path, task['image_path'])
                try:
                    if full_image_path.lower().endswith(('.tif', '.tiff')):
                        image = read_geospatial_image(full_image_path)
                    else:
                        image = Image.open(full_image_path).convert('RGB')
                except FileNotFoundError:
                    print(f"Warning: Image not found at {full_image_path}. Skipping task {task['task_id']}.")
                    batch_msgs.append("IMAGE_NOT_FOUND")
                    continue
                prompt = build_prompt_for_task(task)
                batch_msgs.append([{'role': 'user', 'content': [image, prompt]}])

            valid_batch_tasks = [task for task, msg in zip(batch_tasks, batch_msgs) if msg != "IMAGE_NOT_FOUND"]
            valid_batch_msgs = [msg for msg in batch_msgs if msg != "IMAGE_NOT_FOUND"]
            
            if not valid_batch_msgs:
                continue
            
            # 这部分推理和解析逻辑完全复用
            try:
                raw_outputs = model.chat(
                    image=None, msgs=valid_batch_msgs, tokenizer=tokenizer, max_new_tokens=256
                )
            except Exception as e:
                print("\n" + "="*50)
                print(f"FATAL ERROR during model inference for task '{task_name}' at batch starting index {i}.")
                print(f"Error Details: {e}")
                print("="*50)
                raise e
            
            for task_idx, task in enumerate(valid_batch_tasks):
                raw_output = raw_outputs[task_idx]
                if task['task_type'] == 'choices':
                    choices_list = task['prompt_info']['choices']
                    parsed_answer = extract_characters_regex(raw_output, choices_list)
                else:
                    parsed_answer = str(raw_output).strip()
                results.append({
                    'task_id': task['task_id'], 'task_type': task['task_type'],
                    'model_output': parsed_answer, 'ground_truth': task['ground_truth'],
                    'choices': task['prompt_info'].get('choices', [])
                })
    return results

# MODIFIED: 更新了所有数据和任务相关的参数
def parse_arguments():
    """
    定义并解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="Run inference for remote sensing vision-language tasks.")
    
    # --- 数据路径参数 ---
    parser.add_argument("--caption_json_paths", type=str, nargs='+', default=None,
                        help="One or more full paths to the caption task JSON files.")
    parser.add_argument("--choices_json_paths", type=str, nargs='+', default=None,
                        help="One or more full paths to the multiple-choice task JSON files.")
    parser.add_argument("--images_base_path", type=str,
                        help="[REQUIRED for inference] The base directory path for all image files.")

    # --- 任务控制参数 ---
    parser.add_argument("--tasks_to_run", type=str, nargs='+', default=['all'], choices=['caption', 'choices', 'all'],
                        help="Specify which tasks to run: 'caption', 'choices', or 'all'. Default is 'all'.")
    parser.add_argument("--caption_batch_size", type=int, default="4", help="caption task batch size")
    parser.add_argument("--choices_batch_size", type=int, default="4", help="choices task batch size")
    
    # --- 模式控制参数 ---
    parser.add_argument("--save_path", type=str, default="inference_results", help="Directory to save output results and reports.")
    
    # --- 模型参数 ---
    parser.add_argument("--model_path", type=str, default="None", help="Directory to model.")
    parser.add_argument("--use_hra", action="store_true", help="Use HRA.")
    parser.add_argument("--peft_save_path", type=str, default="None", help="Directory to peft model.")
    
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    # --- 任务选择逻辑 ---
    tasks_to_run = set(args.tasks_to_run)
    if 'all' in tasks_to_run:
        tasks_to_run = {'caption', 'choices'}
    print(f"✅ Tasks selected to run: {list(tasks_to_run)}")

    # --- NEW: 参数校验 ---
    if not args.images_base_path:
        raise ValueError("❌ --images_base_path is required when running model inference.")
    if 'caption' in tasks_to_run and not args.caption_json_paths:
        raise ValueError("❌ Task 'caption' is selected, but --caption_json_paths was not provided.")
    if 'choices' in tasks_to_run and not args.choices_json_paths:
        raise ValueError("❌ Task 'choices' is selected, but --choices_json_paths was not provided.")
    
    # 1. 定义模型与分词器 (逻辑不变)
    model_file = args.model_path
    peft_path, peft_type = None, "Original"
    if args.use_hra: 
        peft_path, peft_type = args.peft_save_path, "HRA"
    # ... (其他模型选择逻辑不变) ...
    print(f"Loading base model from: {model_file}")
    model = AutoModel.from_pretrained(model_file, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
    print("Base model and tokenizer loaded.")
    if peft_path:
        print(f"Loading {peft_type} adapters from: {peft_path}")
        model = PeftModel.from_pretrained(model, peft_path)
        print(f"✅ Successfully added {peft_type} to the model.")
    else:
        print(f"✅ Using {peft_type} model.")

    # 2. MODIFIED: 根据新参数加载数据
    print("Loading test data from specified paths...")
    all_tasks = load_all_test_data(
        tasks_to_run=tasks_to_run,
        caption_files=args.caption_json_paths,
        choices_files=args.choices_json_paths
    )
    if not all_tasks:
        print("❌ No tasks were loaded. Please check your file paths and --tasks_to_run parameter. Exiting.")
        exit()
    print(f"Loaded {len(all_tasks)} tasks in total.")

    # 3. 按任务类型分组并推理 (逻辑不变，但数据源已更新)
    task_configs = {
        'Image caption': {'batch_size': args.caption_batch_size, 'cli_name': 'caption'},
        'choices': {'batch_size': args.choices_batch_size, 'cli_name': 'choices'},
    }
    grouped_tasks = {task_type: [] for task_type in task_configs.keys()}
    for task in all_tasks:
        if task['task_type'] in grouped_tasks:
            grouped_tasks[task['task_type']].append(task)

    all_results = []
    for task_type, config in task_configs.items():
        if config['cli_name'] not in tasks_to_run:
            continue
        
        tasks_for_current_type = grouped_tasks[task_type]
        if not tasks_for_current_type:
            print(f"No tasks for type: {task_type}. Skipping.")
            continue
        
        print(f"\n--- Processing task type: {task_type} ---")
        results_for_current_type = run_batch_inference(
            model=model, tokenizer=tokenizer, tasks=tasks_for_current_type,
            image_base_path=args.images_base_path, # 使用新的图像路径参数
            batch_size=config['batch_size'], task_name=task_type
        )
        all_results.extend(results_for_current_type)
        print(f"--- Finished processing {task_type}. Collected {len(results_for_current_type)} results. ---")

    # 4. 保存结果并计分
    output_dir = args.save_path
    os.makedirs(output_dir, exist_ok=True)
    results_output_path = os.path.join(output_dir, 'inference_results.json')
    with open(results_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll inference results saved to {results_output_path}")

    print("Calculating final scores...")
    score = calculate_final_scores(all_results, tasks_to_run, output_dir)

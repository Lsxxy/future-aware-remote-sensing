import json
import re
import os
import numpy as np
from tqdm import tqdm
import torch
import argparse
from PIL import Image

from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from utils import extract_characters_regex, calculate_bleu_scores, calculate_iou_accuracy, calculate_vqa_accuracy, calculate_mme_rs_accuracy


def load_all_test_data(tasks_to_run, vrs_caption_files=None, vrs_referring_files=None, vrs_vqa_files=None, mme_vqa_files=None):
    """
    根据指定的任务列表和直接的文件路径列表加载测试数据。
    """
    tasks = []
    
    # --- VRSBench 数据加载 ---
    if 'vrs_caption' in tasks_to_run and vrs_caption_files:
        print("Loading VRSBench Caption tasks...")
        for file_path in vrs_caption_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for item in json.load(f):
                        tasks.append({'task_id': item['question_id'], 'image_path': item['image_id'], 'task_type': 'vrs_caption', 'prompt_info': {'question': item['question']}, 'ground_truth': item['ground_truth']})
            except Exception as e: print(f"❌ ERROR reading {file_path}: {e}")

    if 'vrs_referring' in tasks_to_run and vrs_referring_files:
        print("Loading VRSBench Referring tasks...")
        for file_path in vrs_referring_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for item in json.load(f):
                        tasks.append({'task_id': item['question_id'], 'image_path': item['image_id'], 'task_type': 'vrs_referring', 'prompt_info': {'description': item['question']}, 'ground_truth': item['ground_truth']})
            except Exception as e: print(f"❌ ERROR reading {file_path}: {e}")

    if 'vrs_vqa' in tasks_to_run and vrs_vqa_files:
        print("Loading VRSBench VQA tasks...")
        for file_path in vrs_vqa_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for item in json.load(f):
                        tasks.append({'task_id': item['question_id'], 'image_path': item['image_id'], 'task_type': 'vrs_vqa', 'prompt_info': {'question': item['question']}, 'ground_truth': item['ground_truth']})
            except Exception as e: print(f"❌ ERROR reading {file_path}: {e}")

    # --- MME-RealWorld-RS 数据加载 ---
    if 'mme_vqa' in tasks_to_run and mme_vqa_files:
        print("Loading MME-RealWorld-RS VQA tasks...")
        for file_path in mme_vqa_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for item in json.load(f):
                        if item.get('Subtask') == 'Remote Sensing':
                            tasks.append({'task_id': item['Question_id'], 'image_path': item['Image'], 'task_type': 'mme_vqa', 'prompt_info': {'question': item['Text'], 'choices': item['Answer choices']}, 'ground_truth': item['Ground truth']})
            except Exception as e: print(f"❌ ERROR reading {file_path}: {e}")
            
    return tasks

def build_prompt_for_task(task):
    prompt = ""
    if task['task_type'] == 'vrs_caption':
        prompt = "Please describe this image in detail."
    elif task['task_type'] == 'vrs_vqa':
        prompt = f"Answer the following question with a short word or phrase: {task['prompt_info']['question']}"
    elif task['task_type'] == 'vrs_referring':
        prompt = f"What is the bounding box for the object described as \"{task['prompt_info']['description']}\"? Provide the coordinates in the format {{<x_min><y_min><x_max><y_max>}}."
    elif task['task_type'] == 'mme_vqa':
        choices_str = "\n".join(task['prompt_info']['choices'])
        sys_prompt = "Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option."
        prompt = f"{task['prompt_info']['question']}\n{choices_str}\n{sys_prompt}\nThe best answer is:"
    return prompt

# MODIFIED: 函数现在接受 tasks_to_run 以进行条件计分
def calculate_final_scores(results, tasks_to_run, save_path=None):
    # 初始化所有分数和指标
    S1, S2, S_final = 0, 0, 0
    X1, X2, X3, X1_mme = 0, 0, 0, 0
    bleu_results = {}
    report_data = {"overall_scores": {}, "task_details": {}}
    
    tasks_in_vrs = {'vrs_caption', 'vrs_referring', 'vrs_vqa'}
    
    # --- VRSBench 分数计算 ---
    if any(task in tasks_to_run for task in tasks_in_vrs):
        report_data["task_details"]["VRSBench_details"] = {"score": 0, "components": {}}
        
        if 'vrs_caption' in tasks_to_run:
            res = [r for r in results if r['task_type'] == 'vrs_caption']
            if res:
                bleu_results = calculate_bleu_scores([r['ground_truth'] for r in res], [r['model_output'] for r in res])
                X1 = bleu_results.get('Avg_BLEU', 0)
                report_data["task_details"]["VRSBench_details"]["components"]["caption"] = {"metric": "Avg_BLEU", "value": round(X1, 4), "score_contribution": round(X1 * 25, 4), "bleu_details": {k: round(v, 4) for k, v in bleu_results.items()}}
        
        if 'vrs_referring' in tasks_to_run:
            res = [r for r in results if r['task_type'] == 'vrs_referring']
            if res:
                X2 = calculate_iou_accuracy([r['ground_truth'] for r in res], [r['model_output'] for r in res], iou_threshold=0.5)
                report_data["task_details"]["VRSBench_details"]["components"]["referring"] = {"metric": "Accuracy@IoU=0.5", "value": round(X2, 4), "score_contribution": round(X2 * 25, 4)}

        if 'vrs_vqa' in tasks_to_run:
            res = [r for r in results if r['task_type'] == 'vrs_vqa']
            if res:
                X3 = calculate_vqa_accuracy([r['ground_truth'] for r in res], [r['model_output'] for r in res])
                report_data["task_details"]["VRSBench_details"]["components"]["vqa"] = {"metric": "Accuracy", "value": round(X3, 4), "score_contribution": round(X3 * 50, 4)}
        
        S1 = X1 * 25 + X2 * 25 + X3 * 50
        report_data["overall_scores"]["S1_VRSBench"] = round(S1, 4)
        report_data["task_details"]["VRSBench_details"]["score"] = round(S1, 4)
        
    # --- MME-RealWorld-RS 分数计算 ---
    if 'mme_vqa' in tasks_to_run:
        res = [r for r in results if r['task_type'] == 'mme_vqa']
        if res:
            X1_mme = calculate_mme_rs_accuracy([r['ground_truth'] for r in res], [r['model_output'] for r in res])
            S2 = X1_mme * 100
            report_data["overall_scores"]["S2_MME_RealWorld_RS"] = round(S2, 4)
            report_data["task_details"]["MME_RealWorld_RS_details"] = {"score": round(S2, 4), "components": {"vqa": {"metric": "Accuracy", "value": round(X1_mme, 4), "score_contribution": round(S2, 4)}}}

    if S1 > 0 and S2 == 0: S_final = S1
    elif S2 > 0 and S1 == 0: S_final = S2
    elif S1 > 0 and S2 > 0: S_final = (S1 + S2) / 2
    
    report_data["overall_scores"]["S_final_combined"] = round(S_final, 4)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        print(f"Final scores report saved to {os.path.join(save_path, 'results.json')}")

    print("\n" + "="*40 + "\n------ Performance Report ------\n" + "="*40)
    if S1 > 0:
        print(f"VRSBench Score (S1): {S1:.2f} / 100.0")
        if 'vrs_caption' in tasks_to_run: print(f"  - Caption (X1_avg_bleu = {X1:.3f}): {X1 * 25:.2f} / 25")
        if 'vrs_referring' in tasks_to_run: print(f"  - Referring (X2_acc@0.5 = {X2:.3f}): {X2 * 25:.2f} / 25")
        if 'vrs_vqa' in tasks_to_run: print(f"  - VQA (X3_acc = {X3:.3f}): {X3 * 50:.2f} / 50")
        print("-" * 20)
    if S2 > 0:
        print(f"MME-RealWorld-RS Score (S2): {S2:.2f} / 100.0")
        print(f"  - VQA Accuracy (X1_mme = {X1_mme:.3f})")
        print("-" * 20)
    print(f"Final Combined Score (S): {S_final:.2f} / 100.0")
    print("="*40)
    return S_final

def run_batch_inference(model, tokenizer, tasks, image_base_path_dict, batch_size=8, task_name=""):
    results = []
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        progress_bar_desc = f"Running Inference for {task_name}"
        for i in tqdm(range(0, len(tasks), batch_size), desc=progress_bar_desc):
            batch_tasks = tasks[i:i+batch_size]
            batch_msgs = []
            for task in batch_tasks:
                # 动态选择图片基础路径
                img_base_path = image_base_path_dict['mme'] if task['task_type'] == 'mme_vqa' else image_base_path_dict['vrs']
                full_image_path = os.path.join(img_base_path, task['image_path'])
                try:
                    image = Image.open(full_image_path).convert('RGB')
                except FileNotFoundError:
                    print(f"Warning: Image not found at {full_image_path}. Skipping.")
                    batch_msgs.append("IMAGE_NOT_FOUND"); continue
                prompt = build_prompt_for_task(task)
                batch_msgs.append([{'role': 'user', 'content': [image, prompt]}])

            valid_batch_tasks = [task for task, msg in zip(batch_tasks, batch_msgs) if msg != "IMAGE_NOT_FOUND"]
            valid_batch_msgs = [msg for msg in batch_msgs if msg != "IMAGE_NOT_FOUND"]
            if not valid_batch_msgs: continue
            
            try:
                raw_outputs = model.chat(image=None, msgs=valid_batch_msgs, tokenizer=tokenizer, max_new_tokens=256)
            except Exception as e: print(f"\nFATAL ERROR during model inference: {e}"); raise e
            
            for task_idx, task in enumerate(valid_batch_tasks):
                raw_output = raw_outputs[task_idx]
                if task['task_type'] == 'mme_vqa':
                    parsed_answer = extract_characters_regex(raw_output, task['prompt_info']['choices'])
                elif task['task_type'] == 'vrs_referring':
                    match = re.search(r'\{?<?(\d+)>?<?(\d+)>?<?(\d+)>?<?(\d+)>?\}?', str(raw_output))
                    parsed_answer = f"{{<{match.group(1)}><{match.group(2)}><{match.group(3)}><{match.group(4)}>}}" if match else "PARSE_ERROR"
                else:
                    parsed_answer = str(raw_output).strip()
                results.append({'task_id': task['task_id'], 'task_type': task['task_type'], 'model_output': parsed_answer, 'ground_truth': task['ground_truth'], 'choices': task['prompt_info'].get('choices', [])})
    return results

# MODIFIED: 全新的参数解析函数
def parse_arguments():
    """定义并解析所有命令行参数。"""
    parser = argparse.ArgumentParser(description="Run inference for VRSBench and MME-RealWorld-RS datasets.")
    
    # --- 数据路径参数 ---
    parser.add_argument("--vrs_images_path", type=str, help="Base directory for VRSBench images.")
    parser.add_argument("--mme_images_path", type=str, help="Base directory for MME-RealWorld-RS images.")
    parser.add_argument("--vrs_caption_jsons", type=str, nargs='+', help="Path(s) to VRSBench caption JSON file(s).")
    parser.add_argument("--vrs_referring_jsons", type=str, nargs='+', help="Path(s) to VRSBench referring JSON file(s).")
    parser.add_argument("--vrs_vqa_jsons", type=str, nargs='+', help="Path(s) to VRSBench VQA JSON file(s).")
    parser.add_argument("--mme_vqa_jsons", type=str, nargs='+', help="Path(s) to MME-RealWorld-RS VQA JSON file(s).")

    # --- 任务与执行控制参数 ---
    parser.add_argument("--tasks_to_run", type=str, nargs='+', default=['all'], choices=['vrs_caption', 'vrs_referring', 'vrs_vqa', 'mme_vqa', 'all'], help="Specify which tasks to run.")
    parser.add_argument("--vrs_caption_batch_size", type=int, default=20, help="Batch size for VRSBench caption task.")
    parser.add_argument("--vrs_referring_batch_size", type=int, default=20, help="Batch size for VRSBench referring task.")
    parser.add_argument("--vrs_vqa_batch_size", type=int, default=15, help="Batch size for VRSBench VQA task.")
    parser.add_argument("--mme_vqa_batch_size", type=int, default=6, help="Batch size for MME VQA task.")
    
    # --- 模式控制参数 ---
    parser.add_argument("--save_path", type=str, default="inference_results", help="Directory to save output results and reports.")
    
    # --- 模型参数 ---
    parser.add_argument("--model_path", type=str, required=True, help="[REQUIRED] Path to the base model directory.")
    parser.add_argument("--use_hra", action="store_true", help="Flag to use HRA (PEFT) model.")
    parser.add_argument("--peft_save_path", type=str, help="Path to the PEFT model directory (required if --use_hra is set).")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    # --- 任务选择逻辑 ---
    tasks_to_run = set(args.tasks_to_run)
    if 'all' in tasks_to_run:
        tasks_to_run = {'vrs_caption', 'vrs_referring', 'vrs_vqa', 'mme_vqa'}
    print(f"✅ Tasks selected to run: {list(tasks_to_run)}")

    # --- 参数校验 ---
    vrs_tasks = {'vrs_caption', 'vrs_referring', 'vrs_vqa'}
    if any(task in tasks_to_run for task in vrs_tasks) and not args.vrs_images_path:
        raise ValueError("❌ VRSBench task is selected, but --vrs_images_path was not provided.")
    if 'mme_vqa' in tasks_to_run and not args.mme_images_path:
        raise ValueError("❌ MME VQA task is selected, but --mme_images_path was not provided.")

    
    # 1. 定义模型与分词器
    print(f"Loading base model from: {args.model_path}")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print("Base model and tokenizer loaded.")
    if args.use_hra:
        if not args.peft_save_path: raise ValueError("❌ --use_hra is set, but --peft_save_path was not provided.")
        print(f"Loading HRA adapters from: {args.peft_save_path}")
        model = PeftModel.from_pretrained(model, args.peft_save_path)
        print("✅ Successfully added HRA to the model.")

    # 2. 根据参数加载数据
    all_tasks = load_all_test_data(
        tasks_to_run,
        vrs_caption_files=args.vrs_caption_jsons,
        vrs_referring_files=args.vrs_referring_jsons,
        vrs_vqa_files=args.vrs_vqa_jsons,
        mme_vqa_files=args.mme_vqa_jsons
    )
    if not all_tasks: print("❌ No tasks loaded. Exiting."); exit()
    print(f"Loaded {len(all_tasks)} tasks in total.")

    # 3. 动态构建任务配置和图片路径字典
    task_configs = {
        'vrs_caption': {'batch_size': args.vrs_caption_batch_size, 'cli_name': 'vrs_caption'},
        'vrs_referring': {'batch_size': args.vrs_referring_batch_size, 'cli_name': 'vrs_referring'},
        'vrs_vqa': {'batch_size': args.vrs_vqa_batch_size, 'cli_name': 'vrs_vqa'},
        'mme_vqa': {'batch_size': args.mme_vqa_batch_size, 'cli_name': 'mme_vqa'},
    }
    image_base_path_dict = {'vrs': args.vrs_images_path, 'mme': args.mme_images_path}
    
    # 4. 按任务类型分组
    grouped_tasks = {task_type: [] for task_type in task_configs.keys()}
    for task in all_tasks:
        if task['task_type'] in grouped_tasks:
            grouped_tasks[task['task_type']].append(task)

    # 5. 循环推理
    all_results = []
    for task_type, config in task_configs.items():
        if config['cli_name'] not in tasks_to_run or not grouped_tasks[task_type]:
            continue
        print(f"\n--- Processing task type: {task_type} ---")
        results = run_batch_inference(
            model=model, tokenizer=tokenizer, tasks=grouped_tasks[task_type],
            image_base_path_dict=image_base_path_dict,
            batch_size=config['batch_size'], task_name=task_type
        )
        all_results.extend(results)
        print(f"--- Finished. Collected {len(results)} results. ---")

    # 6. 保存并计分
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'inference_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll inference results saved to {os.path.join(args.save_path, 'inference_results.json')}")



    print("Calculating final scores...")
    score = calculate_final_scores(all_results, tasks_to_run, args.save_path)

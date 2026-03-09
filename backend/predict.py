import os
import re
import sys
import yaml
import json
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64

from transformers import AutoModel, AutoTokenizer
from peft import PeftModel


def extract_characters_regex(model_output, choices):
    match = re.search(r"[A-D]", model_output)
    if match:
        return match.group(0)
    return model_output # 如果没匹配到，返回原始输出

script_directory = os.path.dirname(os.path.abspath(__file__))

# --- 全局变量定义 ---
MODEL_PATH = os.path.join(script_directory, '..',  'model_and_hra', 'model')
PEFT_PATH = os.path.join(script_directory, '..', 'model_and_hra', 'hra_merge_data')    
PROMPT_LIBRARY_PATH = os.path.join(script_directory, '..', 'model_and_hra', 'prompt_library.yaml')


# --- 全局模型对象 ---
model = None
tokenizer = None
PROMPTS = {}

# --- 辅助函数 ---
def get_language(text: str) -> str:
    if text and re.search(r"[一-龥]", text):
        return "zh"
    return "en"

def build_prompt_for_task(task):
    if not PROMPTS: 
        return task['prompt_info'].get('question', '')

    lang = 'en' # 默认语言为英文
    
    # 根据任务类型构建不同的Prompt
    if task['task_type'] == 'Image caption':
        lang = get_language(task.get('ground_truth', ''))
        prompt = PROMPTS[lang]['caption_sys_prompt']
        
    elif task['task_type'] == 'choices':
        # 这里构建选择题的Prompt
        question = task['prompt_info']['question']
        # 确定语言
        lang = get_language(question)
        
        sys_prompt = PROMPTS[lang]['choices_sys_prompt']
        
        # 将选项列表格式化为字符串
        choices_str = "\n".join(task['prompt_info']['choices'])
        
        final_answer_prompt = "最终答案是:" if lang == 'zh' else "The best answer is:"
        prompt = f"{question}\n{choices_str}\n\n{sys_prompt}\n\n{final_answer_prompt}"
        
    else:
        prompt = task['prompt_info'].get('question', '')

    return prompt

# --- 服务启动时的模型加载函数 ---
def load_model():
    global model, tokenizer, PROMPTS
    print("="*50)
    print("🚀 Loading model and tokenizer...")
    try:
        with open(PROMPT_LIBRARY_PATH, 'r', encoding='utf-8') as f:
            PROMPTS = yaml.safe_load(f)
        print("✅ Prompt library loaded successfully.")
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✅ Base model and tokenizer loaded.")
        if PEFT_PATH and os.path.exists(PEFT_PATH):
            model = PeftModel.from_pretrained(model, PEFT_PATH)
            print("✅ Successfully added PEFT adapters.")
        print("🎉 Model is ready!")
        print("="*50)
    except Exception as e:
        print(f"❌ FATAL ERROR during model loading: {e}")
        raise e

# --- Flask 应用定义 ---
app = Flask(__name__)
CORS(app)

# --- Caption 接口---
@app.route('/predict_caption', methods=['POST'])
def predict_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        mock_task = {'task_type': 'Image caption', 'ground_truth': ''} # 传入ground_truth用于语言检测
        prompt = build_prompt_for_task(mock_task)
        msgs = [[{'role': 'user', 'content': [image, prompt]}]]
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            raw_outputs = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, max_new_tokens=256)
        model_output = str(raw_outputs[0]).strip()
        print(model_output)
        return jsonify({"description": model_output})
    except Exception as e:
        print(f"❌ Error during caption prediction: {e}")
        return jsonify({"error": "Error during caption prediction", "details": str(e)}), 500


# --- 多项选择题接口 ---
@app.route('/predict_mcq', methods=['POST'])
def predict_mcq():
    """处理前端发来的多项选择题预测请求"""
    
    # 1. 检查请求中是否包含所需数据
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    if 'json_question' not in request.form:
        return jsonify({"error": "No json_question data provided"}), 400

    image_file = request.files['image']
    question_str = request.form['json_question']

    try:
        # 2. 解析数据
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        question_data = json.loads(question_str) # 将JSON字符串变回Python字典

        # 3. 构建用于推理的输入
        mcq_task = {
            'task_type': 'choices',
            'prompt_info': {
                'question': question_data['Text'],
                'choices': question_data['Answer choices']
            }
        }
        prompt = build_prompt_for_task(mcq_task)
        msgs = [[{'role': 'user', 'content': [image, prompt]}]]

        # 4. 执行模型推理
        print("Processing a new MCQ prediction request...")
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            print("开始推理")
            raw_outputs = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                max_new_tokens=256 
            )
        
        raw_output = str(raw_outputs[0]).strip()
        
        # 5. 从模型原始输出中解析出选项 (A, B, C, D)
        parsed_answer = extract_characters_regex(raw_output, mcq_task['prompt_info']['choices'])
        print(f"Prediction successful. Raw: '{raw_output}', Parsed: '{parsed_answer}'")
        
        # 6. 构造返回给前端的JSON
        return jsonify({"model_answer": parsed_answer})

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format for question_data"}), 400
    except Exception as e:
        print(f"❌ Error during MCQ prediction: {e}")
        return jsonify({"error": "An error occurred during MCQ prediction.", "details": str(e)}), 500

# --- 主程序入口 ---
if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False) 
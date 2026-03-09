import sys
sys.path.insert(0, "../../")
import promptwizard
from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing
from promptwizard.glue.common.utils.file import save_jsonlist
from typing import Any
from tqdm import tqdm
from re import compile, findall
from openai import OpenAI
import os
import re
import json
import pickle 
from datasets import load_dataset
from typing import Any, Dict


from dotenv import load_dotenv
load_dotenv(override = True)

class ImageCaptioningProcessor(DatasetSpecificProcessing):
    """
    专门为图像描述任务设计的处理器。
    它负责解析数据、从模型输出中提取描述，并使用一个强大的"裁判LLM"来评估描述的质量。
    """
    
    def __init__(self, judge_llm_client: Any, **kwargs: Any):
        """
        初始化处理器。

        Args:
            judge_llm_client (Any): 一个已经配置好的、用于调用“裁判LLM”（如GPT-4）的客户端实例。
            **kwargs: 其他从基类传入的参数。
        """
        super().__init__(**kwargs)
        self.judge_llm_client = judge_llm_client
        self.INVALID_ANS = "[无效答案]"
        print("图像描述任务处理器已初始化。")

    def dataset_to_jsonl(self, dataset_json_path: str, output_jsonl_path: str, image_base_path: str, task_type: str) -> None:
        processed_records = []
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            # 构造 answer 字段
            ground_truth = item.get("Ground truth", "")

            # 构造 question 字段
            question_text = ""
            # 构造完整的图片路径，但我们只在最终应用中使用，
            # PW流程中，我们把这个路径信息也嵌入question，方便后续处理
            # 或者在自定义类中处理
            full_image_path = os.path.join(image_base_path, item["Image"])
            
            if task_type == 'caption':
                # 格式: <image>\n[图片路径]\n[问题文本]
                # 把图片路径也放进去，方便自定义类读取
                question_text = f"{item['Text']}"
            
            elif task_type == 'mcq':
                choices_text = "\n".join(item["Answer choices"])
                question_text = (
                    f"<image>\n{full_image_path}\n{item['Text']}\n\n"
                    f"选项如下:\n{choices_text}\n\n"
                    f"请仅回答正确选项的字母 (A, B, C, 或 D)。"
                )
                
            processed_records.append({
                "image": full_image_path,
                "question": question_text,
                "final_answer": ground_truth
            })

        # 保存为.jsonl文件
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for record in processed_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"成功将 {dataset_json_path} 转换为 {output_jsonl_path}")

    def extract_answer_from_output(self, completion: str) -> str:
        """
        从数据集的 'answer' 字段中提取标准答案 (Ground Truth)。
        对于图像描述任务，标准答案就是完整的描述文本。

        Args:
            completion (str): 从 .jsonl 文件中读取的 'answer' 字段的值。

        Returns:
            str: 清理后的标准答案文本。
        """
        if not completion or not isinstance(completion, str):
            return self.INVALID_ANS
        # 对于caption任务，我们直接返回完整的文本，只做简单的清理
        return completion.strip()

    def extract_final_answer(self, model_output: str) -> str:
        """
        从您的“九格-4B”模型的完整输出中，提取出核心的图像描述部分。

        为了实现健壮的提取，强烈建议您在优化Prompt时，指示模型
        将最终的描述包裹在特定的标签中，例如：<CAPTION_START>...<CAPTION_END>。

        Args:
            model_output (str): “九格-4B”模型返回的完整文本。

        Returns:
            str: 提取出的图像描述文本。
        """
        if not model_output:
            return self.INVALID_ANS

        # 策略1：使用正则表达式查找特定标签（推荐，最健壮）
        # 假设您的Prompt指示模型这样做
        match = re.search(r"<CAPTION_START>(.*?)</CAPTION_END>", model_output, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 策略2：如果找不到标签，作为备选方案，直接返回整个输出。
        # 这在探索阶段很有用，但对于生产环境，标签法更好。
        print("警告: 在模型输出中未找到 <CAPTION_START>/<CAPTION_END> 标签。将返回完整输出作为描述。")
        return model_output.strip()

    def access_answer(self, llm_output: str, sample: Dict[str, Any]) -> (str, bool):
        """
        评估模型生成的描述是否“正确”。
        这是通过调用一个强大的“裁判LLM”（如GPT-4）来实现的。

        Args:
            llm_output (str): “九格-4B”模型的完整输出。
            sample (Dict[str, Any]): 来自 .jsonl 文件的当前评估样本，包含 'question' 和 'answer'。

        Returns:
            Tuple[str, bool]: 一个元组，包含(提取出的模型描述, 评估结果的布尔值)。
        """
        # 步骤 1: 从模型输出中提取其生成的描述
        model_caption = self.extract_final_answer(llm_output)
        if model_caption == self.INVALID_ANS:
            return self.INVALID_ANS, False

        # 步骤 2: 从样本中提取标准答案描述
        ground_truth_caption = self.extract_answer_from_output(sample)
        if ground_truth_caption == self.INVALID_ANS:
            # 如果没有标准答案，无法评估
            return model_caption, False

        # 步骤 3: 构建给“裁判LLM”的评估指令 (这是核心！)
        judge_prompt = f"""
        [任务背景]
        你是一个极其严谨和公正的遥感图像描述评估专家。你的任务是比较一个AI模型生成的图像描述和一个权威的参考答案，然后给出一个客观的评价。

        [评估标准]
        1. **内容准确性**: 模型描述是否准确地反映了图像中的关键地物和场景？是否存在明显的错误或幻觉？
        2. **细节完整性**: 与参考答案相比，模型描述是否涵盖了足够多的重要细节？有没有遗漏关键信息？
        3. **语言流畅性**: 描述是否通顺、自然、专业？

        [权威参考答案 (Ground Truth)]
        {ground_truth_caption}

        [模型生成的描述]
        {model_caption}

        [你的任务与输出格式]
        请综合以上标准进行评估，并严格按照以下JSON格式返回你的结论，不要包含任何额外的解释文字。如果Ground Truth和模型生成的描述为英文你就返回英文，如果为中文你就返回中文。
        {{
          "score": <一个1到5的整数评分，5分代表完美，1分代表完全错误>,
          "is_correct": <一个布尔值 (true/false)。当且仅当你的综合评分大于等于3分时，此项为 true，否则为 false>,                                                                             
          "reasoning": "<请用一句话简要说明你打分的主要理由>"
        }}
        """

##当且仅当你的综合评分大于等于1分时，此项为 true，否则为 false>,
        # 步骤 4: 调用“裁判LLM”并解析结果
        try:
            # 确保您的客户端调用方式与此匹配
            # 使用 response_format={"type": "json_object"} 是最新OpenAI API的功能，可以强制输出JSON，非常推荐
            response = self.judge_llm_client.chat.completions.create(
                model=os.environ["OPENAI_MODEL_NAME"],  # 推荐使用最强大的模型作为裁判
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"} 
            )
            judge_response_text = response.choices[0].message.content
            
            # 解析JSON结果
            judge_result = json.loads(judge_response_text)
            is_correct = judge_result.get("is_correct", False)

            # (可选) 打印评估过程用于调试
            print(f"裁判评分: {judge_result.get('score')}, 理由: {judge_result.get('reasoning')}")
            
            return is_correct,model_caption

        except Exception as e:
            print(f"LLM-as-a-Judge 评估过程中发生错误: {e}")
            # 在评估出错时，为了安全起见，我们默认评估为不正确
            return model_caption, False


if os.environ['USE_OPENAI_API_KEY'] == "True":
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
caption = ImageCaptioningProcessor(client)
caption.dataset_to_jsonl("/root/Documents/code/FM9G4B-V/data/valid/en_caption.json", 
                         "/root/Documents/code/FM9G4B-V/PromptWizard/choices/data/en_caption.jsonl", 
                         "/root/Documents/code/FM9G4B-V/data/valid/images", 
                         "caption")
promptopt_config_path = '/root/Documents/code/FM9G4B-V/PromptWizard/caption/configs/promptopt_config.yaml'
setup_config_path = '/root/Documents/code/FM9G4B-V/PromptWizard/caption/configs/setup_config.yaml'
train_file_name = '/root/Documents/code/FM9G4B-V/PromptWizard/caption/data/en_caption.jsonl'
gp = GluePromptOpt(promptopt_config_path,
                   setup_config_path,
                   train_file_name,
                   caption)
best_prompt = gp.get_best_prompt(use_examples=True,run_without_train_examples=False,generate_synthetic_examples=False)
if not os.path.exists("results"):
    os.system("mkdir results")
    
with open("results/best_prompt.pkl", 'wb') as f:
    pickle.dump(best_prompt, f)
# with open("results/expert_profile.pkl", 'wb') as f:
#     pickle.dump(expert_profile, f)

print(f"Best prompt: {best_prompt} ")
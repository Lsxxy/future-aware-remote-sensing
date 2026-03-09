import json

def convert_data_format(input_file_path, output_file_path):
    """
    将原始格式的JSON数据转换为目标格式。

    Args:
        input_file_path (str): 输入的原始JSON文件路径。
        output_file_path (str): 输出的目标格式JSON文件路径。
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：输入文件未找到 -> {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：输入文件不是一个有效的JSON格式 -> {input_file_path}")
        return

    converted_data = []
    # 遍历原始数据中的每一个字典（样本）
    for index, original_item in enumerate(original_data):
        # 创建一个新的字典来存放转换后的数据
        new_item = {
            # 1. 'id' 字段：从0开始的字符串索引
            "id": str(index),
            
            # 2. 'image' 字段：直接从原始数据中复制
            "image": original_item.get("image", ""),
            
            # 3. 'conversations' 字段：需要转换内部结构
            "conversations": []
        }

        # 检查原始的'conversations'是否存在且是一个列表
        if "conversations" in original_item and isinstance(original_item["conversations"], list):
            # 遍历原始对话中的每一轮
            for original_convo in original_item["conversations"]:
                # 确定角色（role）
                if original_convo.get("from") == "human":
                    role = "user"
                elif original_convo.get("from") == "gpt":
                    role = "assistant"
                else:
                    # 如果有其他未知的'from'值，可以跳过或设为默认值
                    continue
                
                # 创建新的对话字典
                new_convo = {
                    "role": role,
                    "content": original_convo.get("value", "")
                }
                new_item["conversations"].append(new_convo)
        
        # 将转换后的新样本添加到最终列表中
        converted_data.append(new_item)

    # 将转换后的数据写入新的JSON文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # json.dump 用于将Python对象写入文件
            # indent=4 让输出的JSON文件格式优美，易于阅读
            # ensure_ascii=False 确保中文字符能被正确写入，而不是被转义成\uXXXX
            json.dump(converted_data, f, indent=4, ensure_ascii=False)
        print(f"转换成功！已将结果保存至 -> {output_file_path}")
    except IOError:
        print(f"错误：无法写入到输出文件 -> {output_file_path}")

# --- 使用示例 ---
# 假设你的原始文件名为 'current_data.json'
# 你希望转换后的文件名为 'target_data.json'

input_filename = '/root/Documents/code_2/FM9G4B-V/data/train/zh_caption_train_2.json'
output_filename = '/root/Documents/code_2/FM9G4B-V/data/train/zh_caption_convert_train.json'

# 调用转换函数
convert_data_format(input_filename, output_filename)
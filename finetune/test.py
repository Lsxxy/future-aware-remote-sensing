import json
import os

def add_path_prefix_to_images(input_file_path, output_file_path, image_path_prefix):
    """
    读取JSON文件，为 'image' 字段的值添加路径前缀，并保存到新文件。

    Args:
        input_file_path (str): 输入的JSON文件路径。
        output_file_path (str): 输出的新JSON文件路径。
        image_path_prefix (str): 要添加到图片名前面的路径前缀。
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：输入文件未找到 -> {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：输入文件不是一个有效的JSON格式 -> {input_file_path}")
        return

    # 遍历数据中的每一个样本（字典）
    for item in data:
        # 检查 'image' 键是否存在并且其值是一个字符串
        if "image" in item and isinstance(item["image"], str):
            # 获取原始的文件名
            original_filename = item["image"]
            
            # 使用 os.path.join 来安全地拼接路径
            # 这会自动处理路径分隔符（在Linux上是'/'），比直接用字符串相加更稳健
            new_image_path = os.path.join(image_path_prefix, original_filename)
            
            # 更新当前样本的 'image' 字段
            item["image"] = new_image_path
        else:
            print(f"警告：在 id='{item.get('id', 'N/A')}' 的样本中未找到 'image' 字段或其值不是字符串，已跳过。")

    # 将修改后的数据写入新的JSON文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # indent=4 让输出的JSON文件格式优美，易于阅读
            # ensure_ascii=False 确保路径中的特殊字符（如果有）被正确处理
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"路径添加成功！已将结果保存至 -> {output_file_path}")
    except IOError:
        print(f"错误：无法写入到输出文件 -> {output_file_path}")

# --- 使用示例 ---

# 1. 定义你的文件路径和要添加的前缀
input_json_file = '/root/Documents/code_2/FM9G4B-V/data/VRSBench/VRSBench_convert_train.json' # 替换成你的输入文件名
output_json_file = '/root/Documents/code_2/FM9G4B-V/data/VRSBench/VRSBench_convert_train_1.json' # 你希望生成的新文件名
image_prefix = '/root/Documents/code_2/FM9G4B-V/data/VRSBench/Images_train' # 你要添加的路径前缀

# 2. 调用转换函数
add_path_prefix_to_images(input_json_file, output_json_file, image_prefix)
# 🛰️ "遥知未来" —— 面向多任务遥感理解的多模态智能系统

> 第十九届"挑战杯"全国大学生课外学术科技作品竞赛 "人工智能+"专项赛 全国决赛入围作品

## 📖 项目简介

本项目面向遥感图像多任务理解场景，基于"九格4B-V"多模态大模型，通过 **HRA（PEFT）参数高效微调**、**高质量遥感指令数据集构建** 和 **PromptWizard 自动化提示优化**，显著提升了模型在遥感图像描述、视觉问答、目标定位等任务上的性能。

同时，项目提供了一套完整的 **前后端演示系统**，支持用户通过 Web 界面进行图像描述和视觉问答的交互体验。

### 🔧 技术栈

- **模型微调**：Python / PyTorch / PEFT / HRA / DeepSpeed
- **提示优化**：PromptWizard / LLM-as-a-Judge
- **数据构建**：Qwen-VL-Max API / 分层结构化 Prompt
- **前端**：React / Ant Design
- **后端**：Python / Flask / PyTorch / Transformers
- **部署**：Docker / Conda

------

## 🏆 核心成果

### PEFT 方法全面对比

在 VRSBench 和 MME-RealWorld-RS 两个基准数据集上，系统性对比了 6 种主流参数高效微调方法：

| 方法                     | Avg-BLEU (caption) | Accuracy@IoU=0.5 (referring) | Accuracy (VQA) | MME Accuracy | Score     |
| ------------------------ | ------------------ | ---------------------------- | -------------- | ------------ | --------- |
| Original（未微调）       | 0.077              | 0.312                        | 0.474          | 0.423        | 34.03     |
| LoRA                     | 0.216              | 0.428                        | 0.650          | 0.388        | 43.70     |
| P-tuning                 | 0.076              | 0.344                        | 0.477          | 0.433        | 34.60     |
| Bone                     | 0.217              | 0.384                        | 0.660          | 0.394        | 43.75     |
| BOFT                     | 0.208              | 0.426                        | 0.646          | 0.367        | 42.54     |
| HRA                      | 0.215              | 0.427                        | 0.650          | 0.400        | 44.03     |
| **HRA+Data（最终方案）** | **0.219**          | **0.545**                    | **0.662**      | **0.377**    | **45.00** |

### 关键结论

- **HRA 综合最优**：虽然单项指标未全部第一，但 HRA 在各任务上均无明显短板，综合得分 44.03 领先所有单一方法
- **数据增强效果显著**：加入自制数据集后（HRA+Data），指代理解准确率从 0.427 跃升至 0.545（+27%），综合得分达到 45.00（+32.2%）
- **P-tuning 效果有限**：在遥感场景下提升极小，说明提示学习范式难以弥补领域偏移

------

## ✨ 系统演示

本项目提供了一套完整的 Web 演示系统，支持图像描述和多模态视觉问答两大功能：

![系统演示](/assets/demo.gif)

### 功能说明

- **智能图像描述**：上传遥感图像（支持 TIF / PNG / JPG），AI 模型自动生成详细的文本描述，涵盖整体场景和关键区域细节
- **多模态视觉问答**：上传遥感图像并输入选择题（JSON 格式），系统结合图像内容推理出答案，支持对象属性识别、空间关系判断、逻辑推理等复杂任务

------

## 💡 创新点

### 1. 方法创新：系统化 PEFT 评估框架

突破了以往局限于单一方法的研究范式，构建了系统化的参数高效微调技术评估框架。通过严格变量控制，在综合基准上对 LoRA、P-tuning、Bone、BOFT、HRA 进行了全面横向比较，发现部分方法存在"偏科"现象，并从"均衡性"视角甄别出最适合复杂多任务场景的方案。

### 2. 数据创新：自动化遥感指令数据集构建

提出了"自动化生成 + 人工校验"的高质量遥感指令数据集构建流程：

- **数据源**：基于 DOTA-v2（目标检测，18类，180万+标注实例）和 FAIR1M（细粒度识别，100万+标注实例）
- **自动生成**：使用 Qwen-VL-Max 多模态大模型，通过分层结构化 Prompt 批量生成图像描述和选择题数据
- **Prompt 设计**：从全局场景概述、目标类别与属性、细节补充与背景信息、语言风格与约束四个维度引导生成
- **人工校验**：建立涵盖语义准确性、语言规范性、多样性与难度梯度的三维度校验体系

### 3. 工程创新：自动化提示工程

首次将 PromptWizard 自动化提示工程引入复杂遥感多任务场景，替代人工提示设计。针对不同测试集采用差异化策略：开源测试集使用简洁任务导向型提示（避免分布偏移），闭源测试集通过 PromptWizard 自动搜索最优提示。

------

## 🔬 HRA 微调（finetune）

### 方法概述

HRA（Householder Reflection Adaptation）是一种基于 Householder 反射变换的参数高效微调方法。与 LoRA 的加性更新策略（W'=W+ΔW）不同，HRA 通过学习一组 Householder 向量来构造正交矩阵，对预训练权重施加乘法变换（W'=Q*W），以极少参数实现高效微调，同时正交变换的特性能更好地保持原始模型的知识结构。

### 运行微调

```bash
cd finetune

# 使用 DeepSpeed 启动训练（请先修改 finetune_ds.sh 中的模型和数据路径）
bash finetune_ds.sh
```

### 微调后验证

```bash
python test.py --model_path /path/to/base/model --peft_path /path/to/hra_output
```

------

## 🧠 PromptWizard 自动化提示优化

本项目引入 [PromptWizard](https://github.com/microsoft/PromptWizard) 自动化提示工程框架，替代人工提示设计，针对不同遥感子任务自动生成定制化指令。

### 工作原理

1. **定义任务配置**：在 `prompt_library.yaml` 中定义系统提示、任务模板和行为准则
2. **准备评测数据**：将遥感数据集转换为 PromptWizard 所需的 JSONL 格式
3. **自动化优化**：通过遗传算法进行提示变异，结合 LLM-as-a-Judge 评估机制迭代搜索最优提示
4. **输出最佳提示**：生成的最优提示保存在 `results/` 中，可直接用于推理

### 差异化策略

- **开源测试集**：使用简洁任务导向型提示，避免因训练集与测试集分布相同导致的分布偏移
- **闭源测试集**：在图像描述和选择题任务中均应用 PromptWizard 自动搜索最优提示

### 运行示例

```bash
cd prompt_engineering/caption

# 1. 配置环境变量
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL_NAME="qwen-vl-max"
export USE_OPENAI_API_KEY="True"

# 2. 修改 caption_demo.py 中的数据路径，然后运行
python caption_demo.py
```

------

## 🚀 快速开始

### 演示系统运行

#### 环境要求

- Linux 系统（推荐 Ubuntu 20.04+）
- NVIDIA GPU，推荐 16GB+ 显存
- 已安装 NVIDIA 驱动及 CUDA
- 已安装 Conda（Anaconda 或 Miniconda）

#### 启动步骤

```bash
# 1. 克隆仓库
git clone https://gitee.com/Lsyuye/rs-multi-task-system.git
cd RS-MultiTask-System

# 2. 安装前端依赖
cd frontend
npm install
cd ..

# 3. 运行启动脚本
chmod +x scripts/start_linux.sh
./scripts/start_linux.sh

# 4. 等待 1-2 分钟，浏览器自动打开 http://localhost:3000
```

#### 关闭系统

在启动脚本所在终端按 `Ctrl + C` 即可关闭所有服务。

------

### 模型评测运行（Docker 方式）

#### 1. 准备文件

```
workspace/
├── docker_image.zip          # Docker 镜像（需另行获取）
├── model_and_hra/
│   ├── model/                # 基础模型权重
│   └── hra_merge_data/       # HRA 微调权重
└── your_data_folder/         # 评测数据集
```

#### 2. 加载镜像并启动容器

```bash
unzip docker_image.zip
docker load -i fm9g.tar

docker run -it --rm --gpus all \
  -v /path/to/workspace/model_and_hra:/app/model \
  -v /path/to/workspace/your_data_folder:/app/data \
  fm9g /bin/bash
```

⚠️ **注意**：模型权重文件（约数 GB）和 Docker 镜像未包含在本仓库中。

#### 3. 执行评测

```bash
cd /app/inference

python test_VRS_MME.py \
    --model_path /app/model/model \
    --tasks_to_run all \
    --vrs_caption_jsons /app/data/VRSBench/VRSBench_EVAL_Cap.json \
    --vrs_referring_jsons /app/data/VRSBench/VRSBench_EVAL_referring.json \
    --vrs_vqa_jsons /app/data/VRSBench/VRSBench_EVAL_vqa.json \
    --mme_vqa_jsons /app/data/MME/MME_RealWorld.json \
    --vrs_images_path /app/data/VRSBench/Images_val \
    --mme_images_path /app/data/MME \
    --use_hra \
    --peft_save_path /app/model/hra_merge_data \
    --save_path ./output
```

------

## 📦 模型权重获取

由于模型权重文件体积较大，未包含在本仓库中。如需获取，请通过以下方式联系：

- 📧 邮箱：a1277707261@163.com

------

## 📊 评测输出说明

运行评测脚本后，在 `--save_path` 指定目录下会生成：

- `inference_results.json`：每个测试样本的详细预测结果
- `results.json`：各任务汇总分数及加权总分

------

## 📜 License

本项目仅用于学术研究和学习交流，请勿用于商业用途。


[中文文档](./README_zh_CN.md)

<div align="center">

<img src="./assets/logo.png" alt="HunyuanImage-3.0 Logo" width="600">

# 🎨 HunyuanImage-3.0: A Powerful Native Multimodal Model for Image Generation

</div>


<div align="center">

<img src="./assets/banner.png" alt="HunyuanImage-3.0 Banner" width="800">

</div>


<div align="center">
  <a href=https://hunyuan.tencent.com/image target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/tencent/HunyuanImage-3.0 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://github.com/Tencent-Hunyuan/HunyuanImage-3.0 target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px></a>
  <a href=./assets/HunyuanImage_3_0.pdf target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
</div>

<p align="center">
    👏 Join our <a href="./assets/WECHAT.md" target="_blank">WeChat</a> and <a href="https://discord.gg/ehjWMqF5wY">Discord</a> | 
💻 <a href="https://hunyuan.tencent.com/modelSquare/home/play?modelId=289&from=/visual">Official website(官网) Try our model!</a>&nbsp&nbsp
</p>

## 🔥🔥🔥 News
- **September 28, 2025**: 📖 **HunyuanImage-3.0 Technical Report Released** - Comprehensive technical documentation now available
- **September 28, 2025**: 🚀 **HunyuanImage-3.0 Open Source Release** - Inference code and model weights publicly available


## 🧩 Community Contributions

If you develop/use HunyuanImage-3.0 in your projects, welcome to let us know.

## 📑 Open-source Plan

- HunyuanImage-3.0 (Image Generation Model)
  - [x] Inference 
  - [x] HunyuanImage-3.0 Checkpoints
  - [ ] HunyuanImage-3.0-Instruct Checkpoints (with reasoning)
  - [ ] VLLM Support
  - [ ] Distilled Checkpoints
  - [ ] Image-to-Image Generation
  - [ ] Multi-turn Interaction


## 🗂️ Contents
- [🔥🔥🔥 News](#-news)
- [🧩 Community Contributions](#-community-contributions)
- [📑 Open-source Plan](#-open-source-plan)
- [📖 Introduction](#-introduction)
- [✨ Key Features](#-key-features)
- [🛠️ Dependencies and Installation](#-dependencies-and-installation)
  - [💻 System Requirements](#-system-requirements)
  - [📦 Environment Setup](#-environment-setup)
  - [📥 Install Dependencies](#-install-dependencies)
  - [Performance Optimizations](#performance-optimizations)
- [🚀 Usage](#-usage)
  - [🔥 Quick Start with Transformers](#-quick-start-with-transformers)
  - [🏠 Local Installation & Usage](#-local-installation--usage)
  - [🎨 Interactive Gradio Demo](#-interactive-gradio-demo)
- [🧱 Models Cards](#-models-cards)
- [📝 Prompt Guide](#-prompt-guide)
  - [Manually Writing Prompts](#manually-writing-prompts)
  - [System Prompt For Automatic Rewriting the Prompt](#system-prompt-for-automatic-rewriting-the-prompt)
  - [Advanced Tips](#advanced-tips)
  - [More Cases](#more-cases)
- [📊 Evaluation](#-evaluation)
- [📚 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)
- [🌟🚀  Github Star History](#-github-star-history)

---

## 📖 Introduction

**HunyuanImage-3.0** is a groundbreaking native multimodal model that unifies multimodal understanding and generation within an autoregressive framework. Our text-to-image module achieves performance **comparable to or surpassing** leading closed-source models.


<div align="center">
  <img src="./assets/framework.png" alt="HunyuanImage-3.0 Framework" width="90%">
</div>

## ✨ Key Features

* 🧠 **Unified Multimodal Architecture:** Moving beyond the prevalent DiT-based architectures, HunyuanImage-3.0 employs a unified autoregressive framework. This design enables a more direct and integrated modeling of text and image modalities, leading to surprisingly effective and contextually rich image generation.

* 🏆 **The Largest Image Generation MoE Model:** This is the largest open-source image generation Mixture of Experts (MoE) model to date. It features 64 experts and a total of 80 billion parameters, with 13 billion activated per token, significantly enhancing its capacity and performance.

* 🎨 **Superior Image Generation Performance:** Through rigorous dataset curation and advanced reinforcement learning post-training, we've achieved an optimal balance between semantic accuracy and visual excellence. The model demonstrates exceptional prompt adherence while delivering photorealistic imagery with stunning aesthetic quality and fine-grained details.

* 💭 **Intelligent World-Knowledge Reasoning:** The unified multimodal architecture endows HunyuanImage-3.0 with powerful reasoning capabilities. It leverages its extensive world knowledge to intelligently interpret user intent, automatically elaborating on sparse prompts with contextually appropriate details to produce superior, more complete visual outputs.


## 🛠️ Dependencies and Installation

### 💻 System Requirements

* 🖥️ **Operating System:** Linux
* 🎮 **GPU:** NVIDIA GPU with CUDA support
* 💾 **Disk Space:** 170GB for model weights
* 🧠 **GPU Memory:** ≥3×80GB (4×80GB recommended for better performance)

### 📦 Environment Setup

* 🐍 **Python:** 3.12+ (recommended and tested)
* 🔥 **PyTorch:** 2.7.1
* ⚡ **CUDA:** 12.8

### 📥 Install Dependencies

```bash
# 1. First install PyTorch (CUDA 12.8 Version)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# 2. Then install tencentcloud-sdk
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python

# 3. Then install other dependencies
pip install -r requirements.txt
```

#### Performance Optimizations

For **up to 3x faster inference**, install these optimizations:

```bash
# FlashAttention for faster attention computation
pip install flash-attn==2.8.3 --no-build-isolation

# FlashInfer for optimized moe inference. v0.3.1 is tested.
pip install flashinfer-python
```
> 💡**Installation Tips:** It is critical that the CUDA version used by PyTorch matches the system's CUDA version. 
> FlashInfer relies on this compatibility when compiling kernels at runtime. Pytorch 2.7.1+cu128 is tested.
> GCC version >=9 is recommended for compiling FlashAttention and FlashInfer.

> ⚡ **Performance Tips:** These optimizations can significantly speed up your inference!

> 💡**Notation:** When FlashInfer is enabled, the first inference may be slower (about 10 minutes) due to kernel compilation. Subsequent inferences on the same machine will be much faster.

## 🚀 Usage

### 🔥 Quick Start with Transformers

#### 1️⃣ Download model weights

```bash
# Download from HuggingFace and rename the directory.
# Notice that the directory name should not contain dots, which may cause issues when loading using Transformers.
hf download tencent/HunyuanImage-3.0 --local-dir ./HunyuanImage-3
```

#### 2️⃣ Run with Transformers

```python
from transformers import AutoModelForCausalLM

# Load the model
model_id = "./HunyuanImage-3"
# Currently we can not load the model using HF model_id `tencent/HunyuanImage-3.0` directly 
# due to the dot in the name.

kwargs = dict(
    attn_implementation="sdpa",     # Use "flash_attention_2" if FlashAttention is installed
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    moe_impl="eager",   # Use "flashinfer" if FlashInfer is installed
)

model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
model.load_tokenizer(model_id)

# generate the image
prompt = "A brown and white dog is running on the grass"
image = model.generate_image(prompt=prompt, stream=True)
image.save("image.png")
```

### 🏠 Local Installation & Usage

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanImage-3.0.git
cd HunyuanImage-3.0/
```

#### 2️⃣ Download Model Weights

```bash
# Download from HuggingFace
hf download tencent/HunyuanImage-3.0 --local-dir ./HunyuanImage-3
```

#### 3️⃣ Run the Demo
The Pretrain Checkpoint does not automatically rewrite or enhance input prompts, for optimal results currently, we recommend community partners to use deepseek to rewrite the prompts.

```bash
# set env
export DEEPSEEK_KEY_ID="your_deepseek_key_id"
export DEEPSEEK_KEY_SECRET="your_deepseek_key_secret"

python3 run_image_gen.py --model-id ./HunyuanImage-3 --verbose 1 --sys-deepseek-prompt "universal" --prompt "A brown and white dog is running on the grass"
```

#### 4️⃣ Command Line Arguments

| Arguments            | Description                                                     | Default     |
|----------------------|-----------------------------------------------------------------|-------------|
| `--prompt`           | Input prompt                                                    | (Required)  |
| `--model-id`         | Model path                                                      | (Required)  |
| `--attn-impl`        | Attention implementation. Either `sdpa` or `flash_attention_2`. | `sdpa`      |
| `--moe-impl`         | MoE implementation. Either `eager` or `flashinfer`              | `eager`     |
| `--seed`             | Random seed for image generation                                | `None`      |
| `--diff-infer-steps` | Diffusion infer steps                                           | `50`        |
| `--image-size`       | Image resolution. Can be `auto`, like `1280x768` or `16:9`      | `auto`      |
| `--save`             | Image save path.                                                | `image.png` |
| `--verbose`          | Verbose level. 0: No log; 1: log inference information.         | `0`         |

### 🎨 Interactive Gradio Demo

Launch an interactive web interface for easy text-to-image generation.

#### 1️⃣ Install Gradio

```bash
pip install gradio>=4.21.0
```

#### 2️⃣ Configure Environment

```bash
# Set your model path
export MODEL_ID="path/to/your/model"

# Optional: Configure GPU usage (default: 0,1,2,3)
export GPUS="0,1,2,3"

# Optional: Configure host and port (default: 0.0.0.0:443)
export HOST="0.0.0.0"
export PORT="443"
```

#### 3️⃣ Launch the Web Interface

**Basic Launch:**
```bash
sh run_app.sh
```

**With Performance Optimizations:**
```bash
# Use both optimizations for maximum performance
sh run_app.sh --moe-impl flashinfer --attn-impl flash_attention_2
```

#### 4️⃣ Access the Interface

> 🌐 **Web Interface:** Open your browser and navigate to `http://localhost:443` (or your configured port)


## 🧱 Models Cards

| Model                     | Params | Download | Recommended VRAM | Supported |
|---------------------------| --- | --- | --- | --- |
| HunyuanImage-3.0          | 80B total (13B active) | [HuggingFace](https://huggingface.co/tencent/HunyuanImage-3.0) | ≥ 3 × 80 GB | ✅ Text-to-Image
| HunyuanImage-3.0-Instruct | 80B total (13B active) | [HuggingFace](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct) | ≥ 3 × 80 GB | ✅ Text-to-Image<br>✅ Prompt Self-Rewrite <br>✅ CoT Think



Notes:
- Install performance extras (FlashAttention, FlashInfer) for faster inference.
- Multi‑GPU inference is recommended for the Base model.


## 📝 Prompt Guide

### Manually Writing Prompts.
The Pretrain Checkpoint does not automatically rewrite or enhance input prompts, Instruct Checkpoint can rewrite or enhance input prompts with thinking . For optimal results currently, we recommend community partners consulting our official guide on how to write effective prompts.

Reference: [HunyuanImage 3.0 Prompt Handbook](
https://docs.qq.com/doc/DUVVadmhCdG9qRXBU)


### System Prompt For Automatic Rewriting the Prompt.

We've included two system prompts in the PE folder of this repository that leverage DeepSeek to automatically enhance user inputs:

* **system_prompt_universal**: This system prompt converts photographic style, artistic prompts into a detailed one.
* **system_prompt_text_rendering**: This system prompt converts UI/Poster/Text Rending prompts to a deailed on that suits the model.

Note that these system prompts are in Chinese because Deepseek works better with Chinese system prompts. If you want to use it for English oriented model, you may translate it into English or refer to the comments in the PE file as a guide.

We also create a [Yuanqi workflow](https://yuanqi.tencent.com/agent/H69VgtJdj3Dz) to implent the universal one, you can directly try it.

### Advanced Tips
- **Content Priority**: Focus on describing the main subject and action first, followed by details about the environment and style. A more general description framework is: **Main subject and scene + Image quality and style + Composition and perspective + Lighting and atmosphere + Technical parameters**. Keywords can be added both before and after this structure.

- **Image resolution**: Our model not only supports multiple resolutions but also offers both **automatic and specified resolution** options. In auto mode, the model automatically predicts the image resolution based on the input prompt. In specified mode (like traditional DiT), the model outputs an image resolution that strictly aligns with the user's chosen resolution.

### More Cases
Our model can follow complex instructions to generate high‑quality, creative images.

<div align="center">
  <img src="./assets/banner_all.jpg" width=100% alt="HunyuanImage 3.0 Demo">
</div>

Our model can effectively process very long text inputs, enabling users to precisely control the finer details of generated images. Extended prompts allow for intricate elements to be accurately captured, making it ideal for complex projects requiring precision and creativity.

<p align="center">
<table>
<thead>
</thead>
<tbody>
<tr>
<td>
<img src="./assets/pg_imgs/image1.png" width=100%><details>
<summary>Show prompt</summary>
A cinematic medium shot captures a single Asian woman seated on a chair within a dimly lit room, creating an intimate and theatrical atmosphere. The composition is focused on the subject, rendered with rich colors and intricate textures that evoke a nostalgic and moody feeling.\n\nThe primary subject is a young Asian woman with a thoughtful and expressive countenance, her gaze directed slightly away from the camera. She is seated in a relaxed yet elegant posture on an ornate, vintage armchair. The chair is upholstered in a deep red velvet, its fabric showing detailed, intricate textures and slight signs of wear. She wears a simple, elegant dress in a dark teal hue, the material catching the light in a way that reveals its fine-woven texture. Her skin has a soft, matte quality, and the light delicately models the contours of her face and arms.\n\nThe surrounding room is characterized by its vintage decor, which contributes to the historic and evocative mood. In the immediate background, partially blurred due to a shallow depth of field consistent with a f/2.8 aperture, the wall is covered with wallpaper featuring a subtle, damask pattern. The overall color palette is a carefully balanced interplay of deep teal and rich red hues, creating a visually compelling and cohesive environment. The entire scene is detailed, from the fibers of the upholstery to the subtle patterns on the wall.\n\nThe lighting is highly dramatic and artistic, defined by high contrast and pronounced shadow play. A single key light source, positioned off-camera, projects gobo lighting patterns onto the scene, casting intricate shapes of light and shadow across the woman and the back wall. These dramatic shadows create a strong sense of depth and a theatrical quality. While some shadows are deep and defined, others remain soft, gently wrapping around the subject and preventing the loss of detail in darker areas. The soft focus on the background enhances the intimate feeling, drawing all attention to the expressive subject. The overall image presents a cinematic, photorealistic photography style.
</details>
</td>
<td><img src="./assets/pg_imgs/image2.png" width=100%><details>
<summary>Show prompt</summary>
A cinematic, photorealistic medium shot captures a high-contrast urban street corner, defined by the sharp intersection of light and shadow. The primary subject is the exterior corner of a building, rendered in a low-saturation, realistic style.\n\nThe building wall, which occupies the majority of the frame, is painted a warm orange with a finely detailed, rough stucco texture. Horizontal white stripes run across its surface. The base of the building is constructed from large, rough-hewn stone blocks, showing visible particles and texture. On the left, illuminated side of the building, there is a single window with closed, dark-colored shutters. Adjacent to the window, a simple black pendant lamp hangs from a thin, taut rope, casting a distinct, sharp-edged shadow onto the sunlit orange wall. The composition is split diagonally, with the right side of the building enveloped in a deep brown shadow. At the bottom of the frame, a smooth concrete sidewalk is visible, upon which the dynamic silhouette of a person is captured mid-stride, walking from right to left.\n\nIn the shallow background, the faint, out-of-focus outlines of another building and the bare, skeletal branches of trees are softly visible, contributing to the quiet urban atmosphere and adding a sense of depth to the scene. These elements are rendered with minimal detail to keep the focus on the foreground architecture.\n\nThe scene is illuminated by strong, natural sunlight originating from the upper left, creating a dramatic chiaroscuro effect. This hard light source casts deep, well-defined shadows, producing a sharp contrast between the brightly lit warm orange surfaces and the deep brown shadow areas. The lighting highlights the fine details in the wall texture and stone particles, emphasizing the photorealistic quality. The overall presentation reflects a high-quality photorealistic photography style, infused with a cinematic film noir aesthetic.
</details>
</td>
</tr>
<tr>
<td>
<img src="./assets/pg_imgs/image3.png" width=100%><details>
<summary>Show prompt</summary>
一幅极具视觉张力的杂志封面风格人像特写。画面主体是一个身着古风汉服的人物，构图采用了从肩部以上的超级近距离特写，人物占据了画面的绝大部分，形成了强烈的视觉冲击力。\n\n画面中的人物以一种慵懒的姿态出现，微微倾斜着头部，裸露的一侧肩膀线条流畅。她正用一种妩媚而直接的眼神凝视着镜头，双眼微张，眼神深邃，传递出一种神秘而勾人的气质。人物的面部特征精致，皮肤质感细腻，在特定的光线下，面部轮廓清晰分明，展现出一种古典与现代融合的时尚美感。\n\n整个画面的背景被设定为一种简约而高级的纯红色。这种红色色调深沉，呈现出哑光质感，既纯粹又无任何杂质，为整个暗黑神秘的氛围奠定了沉稳而富有张力的基调。这个纯色的背景有效地突出了前景中的人物主体，使得所有视觉焦点都集中在其身上。\n\n光线和氛围的营造是这幅杂志风海报的关键。一束暗橘色的柔和光线作为主光源，从人物的一侧斜上方投射下来，精准地勾勒出人物的脸颊、鼻梁和肩膀的轮廓，在皮肤上形成微妙的光影过渡。同时，人物的周身萦绕着一层暗淡且低饱和度的银白色辉光，如同清冷的月光，形成一道朦胧的轮廓光。这道银辉为人物增添了几分疏离的幽灵感，强化了整体暗黑风格的神秘气质。光影的强烈对比与色彩的独特搭配，共同塑造了这张充满故事感的特写画面。整体图像呈现出一种融合了古典元素的现代时尚摄影风格。
</details>
</td>
<td>
<img src="./assets/pg_imgs/image4.png" width=100%><details>
<summary>Show prompt</summary>
一幅采用极简俯视视角的油画作品，画面主体由一道居中斜向的红色笔触构成。\n\n这道醒目的红色笔触运用了厚涂技法，颜料堆叠形成了强烈的物理厚度和三维立体感。它从画面的左上角附近延伸至右下角附近，构成一个动态的对角线。颜料表面可以清晰地看到画刀刮擦和笔刷拖曳留下的痕迹，边缘处的颜料层相对较薄，而中央部分则高高隆起，形成了不规则的起伏。\n\n在这道立体的红色颜料之上，巧妙地构建了一处精致的微缩景观。景观的核心是一片模拟红海滩的区域，由细腻的深红色颜料点缀而成，与下方基底的鲜红色形成丰富的层次对比。紧邻着“红海滩”的是一小片湖泊，由一层平滑且带有光泽的蓝色与白色混合颜料构成，质感如同平静无波的水面。湖泊边缘，一小撮芦苇丛生，由几根纤细挺拔的、用淡黄色和棕色颜料勾勒出的线条来表现。一只小巧的白鹭立于芦苇旁，其形态由一小块纯白色的厚涂颜料塑造，仅用一抹精炼的黑色颜料点出其尖喙，姿态优雅宁静。\n\n整个构图的背景是大面积的留白，呈现为一张带有细微凹凸纹理的白色纸质基底，这种极简处理极大地突出了中央的红色笔触及其上的微缩景观。\n\n光线从画面一侧柔和地照射下来，在厚涂的颜料堆叠处投下淡淡的、轮廓分明的阴影，进一步增强了画面的三维立体感和油画质感。整幅画面呈现出一种结合了厚涂技法的现代极简主义油画风格。
</details>
</td>
</tr>
<tr>
<td>
<img src="./assets/pg_imgs/image5.png" width=100%><details>
<summary>Show prompt</summary>
整体画面采用一个二乘二的四宫格布局，以产品可视化的风格，展示了一只兔子在四种不同材质下的渲染效果。每个宫格内都有一只姿态完全相同的兔子模型，它呈坐姿，双耳竖立，面朝前方。所有宫格的背景均是统一的中性深灰色，这种简约背景旨在最大限度地突出每种材质的独特质感。\n\n左上角的宫格中，兔子模型由哑光白色石膏材质构成。其表面平滑、均匀且无反射，在模型的耳朵根部、四肢交接处等凹陷区域呈现出柔和的环境光遮蔽阴影，这种微妙的阴影变化凸显了其纯粹的几何形态，整体感觉像一个用于美术研究的基础模型。\n\n右上角的宫格中，兔子模型由晶莹剔透的无瑕疵玻璃制成。它展现了逼真的物理折射效果，透过其透明的身体看到的背景呈现出轻微的扭曲。清晰的镜面高光沿着其身体的曲线轮廓流动，表面上还能看到微弱而清晰的环境反射，赋予其一种精致而易碎的质感。\n\n左下角的宫格中，兔子模型呈现为带有拉丝纹理的钛金属材质。金属表面具有明显的各向异性反射效果，呈现出冷峻的灰调金属光泽。锐利明亮的高光和深邃的阴影形成了强烈对比，精确地定义了其坚固的三维形态，展现了工业设计般的美感。\n\n右下角的宫格中，兔子模型覆盖着一层柔软浓密的灰色毛绒。根根分明的绒毛清晰可见，创造出一种温暖、可触摸的质地。光线照射在绒毛的末梢，形成柔和的光晕效果，而毛绒内部的阴影则显得深邃而柔软，展现了高度写实的毛发渲染效果。\n\n整个四宫格由来自多个方向的、柔和均匀的影棚灯光照亮，确保了每种材质的细节和特性都得到清晰的展现，没有任何刺眼的阴影或过曝的高光。这张图像以一种高度写实的3D渲染风格呈现，完美地诠释了产品可视化的精髓
</details>
</td>
<td>
<img src="./assets/pg_imgs/image6.png" width=100%><details>
<summary>Show prompt</summary>
由一个两行两列的网格构成，共包含四个独立的场景，每个场景都以不同的艺术风格描绘了一个小男孩（小明）一天中的不同活动。\n\n左上角的第一个场景，以超写实摄影风格呈现。画面主体是一个大约8岁的东亚小男孩，他穿着整洁的小学制服——一件白色短袖衬衫和蓝色短裤，脖子上系着红领巾。他背着一个蓝色的双肩书包，正走在去上学的路上。他位于画面的前景偏右侧，面带微笑，步伐轻快。场景设定在清晨，柔和的阳光从左上方照射下来，在人行道上投下清晰而柔和的影子。背景是绿树成荫的街道和模糊可见的学校铁艺大门，营造出宁静的早晨氛围。这张图片的细节表现极为丰富，可以清晰地看到男孩头发的光泽、衣服的褶皱纹理以及书包的帆布材质，完全展现了专业摄影的质感。\n\n右上角的第二个场景，采用日式赛璐璐动漫风格绘制。画面中，小男孩坐在家中的木质餐桌旁吃午饭。他的形象被动漫化，拥有大而明亮的眼睛和简洁的五官线条。他身穿一件简单的黄色T恤，正用筷子夹起碗里的米饭。桌上摆放着一碗汤和两盘家常菜。背景是一个温馨的室内环境，一扇明亮的窗户透进正午的阳光，窗外是蓝天白云。整个画面色彩鲜艳、饱和度高，角色轮廓线清晰明确，阴影部分采用平涂的色块处理，是典型的赛璐璐动漫风格。\n\n左下角的第三个场景，以细腻的铅笔素描风格呈现。画面描绘了下午在操场上踢足球的小男孩。整个图像由不同灰度的石墨色调构成，没有其他颜色。小男孩身穿运动短袖和短裤，身体呈前倾姿态，右脚正要踢向一个足球，动作充满动感。背景是空旷的操场和远处的球门，用简练的线条和排线勾勒。艺术家通过交叉排线和涂抹技巧来表现光影和体积感，足球上的阴影、人物身上的肌肉线条以及地面粗糙的质感都通过铅笔的笔触得到了充分的展现。这张铅笔画突出了素描的光影关系和线条美感。\n\n右下角的第四个场景，以文森特·梵高的后印象派油画风格进行诠释。画面描绘了夜晚时分，小男孩独自在河边钓鱼的景象。他坐在一块岩石上，手持一根简易的钓鱼竿，身影在深蓝色的夜幕下显得很渺小。整个画面的视觉焦点是天空和水面，天空布满了旋转、卷曲的星云，星星和月亮被描绘成巨大、发光的光团，使用了厚涂的油画颜料（Impasto），笔触粗犷而充满能量。深蓝、亮黄和白色的颜料在画布上相互交织，形成强烈的视觉冲击力。水面倒映着天空中扭曲的光影，整个场景充满了梵高作品中特有的强烈情感和动荡不安的美感。这幅画作是对梵高风格的深度致敬。
</details>
</td>
</tr>
<tr>
<td>
<img src="./assets/pg_imgs/image7.png" width=100%><details>
<summary>Show prompt</summary>
以平视视角，呈现了一幅关于如何用素描技法绘制鹦鹉的九宫格教学图。整体构图规整，九个大小一致的方形画框以三行三列的形式均匀分布在浅灰色背景上，清晰地展示了从基本形状到最终成品的全过程。\n\n第一行从左至右展示了绘画的初始步骤。左上角的第一个画框中，用简洁的铅笔线条勾勒出鹦鹉的基本几何形态：一个圆形代表头部，一个稍大的椭圆形代表身体。右上角有一个小号的无衬线字体数字“1”。中间的第二个画框中，在基础形态上添加了三角形的鸟喙轮廓和一条长长的弧线作为尾巴的雏形，头部和身体的连接处线条变得更加流畅；右上角标有数字“2”。右侧的第三个画框中，进一步精确了鹦鹉的整体轮廓，勾勒出头部顶端的羽冠和清晰的眼部圆形轮廓；右上角标有数字“3”。\n\n第二行专注于结构与细节的添加，描绘了绘画的中期阶段。左侧的第四个画框里，鹦鹉的身体上添加了翅膀的基本形状，同时在身体下方画出了一根作为栖木的横向树枝，鹦鹉的爪子初步搭在树枝上；右上角标有数字“4”。中间的第五个画框中，开始细化翅膀和尾部的羽毛分组，用短促的线条表现出层次感，并清晰地画出爪子紧握树枝的细节；右上角标有数字“5”。右侧的第六个画框里，开始为鹦鹉添加初步的阴影，使用交叉排线的素描技法在腹部、翅膀下方和颈部制造出体积感；右上角标有数字“6”。\n\n第三行则展示了最终的润色与完成阶段。左下角的第七个画框中，素描的排线更加密集，阴影层次更加丰富，羽毛的纹理细节被仔细刻画出来，眼珠也添加了高光点缀，显得炯炯有神；右上角标有数字“7”。中间的第八个画框里，描绘的重点转移到栖木上，增加了树枝的纹理和节疤细节，同时整体调整了鹦鹉身上的光影关系，使立体感更为突出；右上角标有数字“8”。右下角的第九个画框是最终完成图，所有线条都经过了精炼，光影对比强烈，鹦鹉的羽毛质感、木质栖木的粗糙感都表现得淋漓尽致，呈现出一幅完整且细节丰富的素描作品；右上角标有数字“9”。\n\n整个画面的光线均匀而明亮，没有任何特定的光源方向，确保了每个教学步骤的视觉清晰度。整体呈现出一种清晰、有条理的数字插画教程风格。
</details>
</td>
<td>
<img src="./assets/pg_imgs/image8.png" width=100%><details>
<summary>Show prompt</summary>
一张现代平面设计风格的海报占据了整个画面，构图简洁且中心突出。\n\n海报的主体是位于画面正中央的一只腾讯QQ企鹅。这只企鹅采用了圆润可爱的3D卡通渲染风格，身体主要为饱满的黑色，腹部为纯白色。它的眼睛大而圆，眼神好奇地直视前方。黄色的嘴巴小巧而立体，双脚同样为鲜明的黄色，稳稳地站立着。一条标志性的红色围巾整齐地系在它的脖子上，围巾的材质带有轻微的布料质感，末端自然下垂。企鹅的整体造型干净利落，边缘光滑，呈现出一种精致的数字插画质感。\n\n海报的背景是一种从上到下由浅蓝色平滑过渡到白色的柔和渐变，营造出一种开阔、明亮的空间感。在企鹅的身后，散布着一些淡淡的、模糊的圆形光斑和几道柔和的抽象光束，为这个简约的平面设计海报增添了微妙的深度和科技感。\n\n画面的底部区域是文字部分，排版居中对齐。上半部分是一行稍大的黑色黑体字，内容为“Hunyuan Image 3.0”。紧随其下的是一行字号略小的深灰色黑体字，内容为“原生多模态大模型”。两行文字清晰易读，与整体的现代平面设计风格保持一致。\n\n整体光线明亮、均匀，没有明显的阴影，突出了企鹅和文字信息，符合现代设计海报的视觉要求。这张图像呈现了现代、简洁的平面设计海报风格。
</details>
</td>
</tr>
</tbody>
</table>
</p>

## 📊 Evaluation

* 🤖 **SSAE (Machine Evaluation)**   
SSAE (Structured Semantic Alignment Evaluation) is an intelligent evaluation metric for image-text alignment based on advanced multimodal large language models (MLLMs). We extracted 3500 key points across 12 categories, then used multimodal large language models to automatically evaluate and score by comparing the generated images with these key points based on the visual content of the images. Mean Image Accuracy represents the image-wise average score across all key points, while Global Accuracy directly calculates the average score across all key points.

<p align="center">
  <img src="./assets/ssae_side_by_side_comparison.png" width=98% alt="Human Evaluation with Other Models">
</p>

<p align="center">
  <img src="./assets/ssae_side_by_side_heatmap.png" width=98% alt="Human Evaluation with Other Models">
</p>


* 👥 **GSB (Human Evaluation)** 

We adopted the GSB (Good/Same/Bad) evaluation method commonly used to assess the relative performance between two models from an overall image perception perspective. In total, we utilized 1,000 text prompts, generating an equal number of image samples for all compared models in a single run. For a fair comparison, we conducted inference only once for each prompt, avoiding any cherry-picking of results. When comparing with the baseline methods, we maintained the default settings for all selected models. The evaluation was performed by more than 100 professional evaluators. 

<p align="center">
  <img src="./assets/gsb.png" width=98% alt="Human Evaluation with Other Models">
</p>


## 📚 Citation

If you find HunyuanImage-3.0 useful in your research, please cite our work:

```bibtex
@misc{HunyuanImage-3.0,
  title={HunyuanImage 3.0: Technical Report},
  author={Tencent Hunyuan Team},
  year={2025},
  howpublished={\url{https://github.com/Tencent-Hunyuan/HunyuanImage-3.0}},
}
```

## 🙏 Acknowledgements

We extend our heartfelt gratitude to the following open-source projects and communities for their invaluable contributions:

* 🤗 [Transformers](https://github.com/huggingface/transformers) - State-of-the-art NLP library
* 🎨 [Diffusers](https://github.com/huggingface/diffusers) - Diffusion models library  
* 🌐 [HuggingFace](https://huggingface.co/) - AI model hub and community
* ⚡ [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Memory-efficient attention
* 🚀 [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - Optimized inference engine

## 🌟🚀 Github Star History

[![GitHub stars](https://img.shields.io/github/stars/Tencent-Hunyuan/HunyuanImage-3.0?style=social)](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)
[![GitHub forks](https://img.shields.io/github/forks/Tencent-Hunyuan/HunyuanImage-3.0?style=social)](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)


[![Star History Chart](https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-3.0&type=Date)](https://www.star-history.com/#Tencent-Hunyuan/HunyuanImage-3.0&Date)

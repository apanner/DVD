
## Introduction


## News

## Installation

### Install from source code (recommended):


```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DVD
conda create -n DVD python=3.10 -y 
conda activate dvd 
pip install -e .
```

### Install SageAttention (For Speedup):
```
pip install sageattention
```
### Download the checkpoint from Huggingface

```
mkdir ckpt
cd ckpt 
huggingface
```
If you encounter issues during installation, it may be caused by the packages we depend on. Please refer to the documentation of the package that caused the problem.

## Usage (in Python code)

The Python examples are in [`examples`](./examples/). We provide an overview here.

### Download Models

Download the pre-set models. Model IDs can be found in [config file](/diffsynth/configs/model_config.py).

```python
from diffsynth import download_models

download_models(["FLUX.1-dev", "Kolors"])
```

Download your own models.

```python
from diffsynth.models.downloader import download_from_huggingface, download_from_modelscope

# From Modelscope (recommended)
download_from_modelscope("Kwai-Kolors/Kolors", "vae/diffusion_pytorch_model.fp16.bin", "models/kolors/Kolors/vae")
# From Huggingface
download_from_huggingface("Kwai-Kolors/Kolors", "vae/diffusion_pytorch_model.fp16.safetensors", "models/kolors/Kolors/vae")
```

### Video Synthesis

#### Text-to-video using CogVideoX-5B

CogVideoX-5B is released by ZhiPu. We provide an improved pipeline, supporting text-to-video, video editing, self-upscaling and video interpolation. [`examples/video_synthesis`](./examples/video_synthesis/)

The video on the left is generated using the original text-to-video pipeline, while the video on the right is the result after editing and frame interpolation.

https://github.com/user-attachments/assets/26b044c1-4a60-44a4-842f-627ff289d006

#### Long Video Synthesis

We trained extended video synthesis models, which can generate 128 frames. [`examples/ExVideo`](./examples/ExVideo/)

https://github.com/modelscope/DiffSynth-Studio/assets/35051019/d97f6aa9-8064-4b5b-9d49-ed6001bb9acc

https://github.com/user-attachments/assets/321ee04b-8c17-479e-8a95-8cbcf21f8d7e

#### Toon Shading

Render realistic videos in a flatten style and enable video editing features. [`examples/Diffutoon`](./examples/Diffutoon/)

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/20528af5-5100-474a-8cdc-440b9efdd86c

#### Video Stylization

Video stylization without video models. [`examples/diffsynth`](./examples/diffsynth/)

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/59fb2f7b-8de0-4481-b79f-0c3a7361a1ea

### Image Synthesis

Generate high-resolution images, by breaking the limitation of diffusion models! [`examples/image_synthesis`](./examples/image_synthesis/).

LoRA fine-tuning is supported in [`examples/train`](./examples/train/).

|FLUX|Stable Diffusion 3|
|-|-|
|![image_1024_cfg](https://github.com/user-attachments/assets/984561e9-553d-4952-9443-79ce144f379f)|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4df346db-6f91-420a-b4c1-26e205376098)|

|Kolors|Hunyuan-DiT|
|-|-|
|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/53ef6f41-da11-4701-8665-9f64392607bf)|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/60b022c8-df3f-4541-95ab-bf39f2fa8bb5)|

|Stable Diffusion|Stable Diffusion XL|
|-|-|
|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5)|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90)|

## Usage (in WebUI)

Create stunning images using the painter, with assistance from AI!

https://github.com/user-attachments/assets/95265d21-cdd6-4125-a7cb-9fbcf6ceb7b0

**This video is not rendered in real-time.**

Before launching the WebUI, please download models to the folder `./models`. See [here](#download-models).

* `Gradio` version

```
pip install gradio
```

```
python apps/gradio/DiffSynth_Studio.py
```

![20240822102002](https://github.com/user-attachments/assets/59613157-de51-4109-99b3-97cbffd88076)

* `Streamlit` version

```
pip install streamlit streamlit-drawable-canvas
```

```
python -m streamlit run apps/streamlit/DiffSynth_Studio.py
```

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/93085557-73f3-4eee-a205-9829591ef954

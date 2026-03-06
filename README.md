<h2 align="center"> <a href="https://arxiv.org/abs/2410.22108">MTMCS-Bench: Evaluating Contextual Safety of Multimodal Large Language Models in Multi-Turn Dialogues</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

<div align="center">    
<img src="./asset/demo.jpg" width="100%" height="50%">
</div>
 

## Abstract :bulb:
Multimodal large language models (MLLMs) are increasingly deployed as assistants that interact through text and images, 
making it crucial to evaluate contextual safety when risk depends on both the visual scene and the evolving dialogue. 
Existing contextual safety benchmarks are mostly single-turn and often miss how malicious intent can emerge gradually or how the same scene can support both benign and exploitative goals. 
We introduce the Multi-Turn Multimodal Contextual Safety Benchmark (MTMCS-Bench), a benchmark of realistic images and multi-turn conversations that evaluates contextual safety in MLLMs under two complementary settings, escalation-based risk and context-switch risk. 
MTMCS-Bench offers paired safe and unsafe dialogues with structured evaluation. It contains over 30 thousand multimodal (image+text) and unimodal (text-only) samples, with metrics that separately measure contextual intent recognition, safety-awareness on unsafe cases, and helpfulness on benign ones. 
Across eight open-source and seven proprietary MLLMs, we observe persistent trade-offs between contextual safety and utility, with models tending to either miss gradual risks or over-refuse benign dialogues. Finally, we evaluate five current guardrails and find that they mitigate some failures but do not fully resolve multi-turn contextual risks.

## News :mega:
- **[Jan 08, 2026]** Upload benchmark data to HF. The link can be referred to [here](https://huggingface.co/datasets/ND-25/MCS-bench).


## Quick Access :newspaper:
- [Huggingface Dataset](https://huggingface.co/datasets/ND-25/MCS-bench): Our benchmark is available on Huggingface.
- [Arxiv Paper](https://arxiv.org/abs/2410.22108): Detailed information about the MTMCS-Bench and its unique evaluation.
- [GitHub Repository](https://github.com/franciscoliu/MTMCS-Bench): Access the inference/evaluation code and additional resources for the MTMCS-Bench dataset.


## Installation :books:

First, you can install the required packages using requirements.txt

```bash
pip install -r requirements.txt
```
Then, to use the OpenAI and Claude APIs, you need to set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export CLAUDE_API_KEY="your_claude_api_key"
``` 

## Inference :wrench:

We currently support 15 models in total; you can find the full list of supported models in the `model` folder. You need to replace "DATASET_DIR" to the actual benchmark name at `inference.py` file. 
To run inference with a model, use the following command (after you replace the dataset with the correct name):

```bash
python inference.py \
	--model_type MODEL_TYPE \
	--model_path MODEL_PATH \
	--split both \
	--output_root OUTPUT_ROOT
```

Here, `MODEL_TYPE` is the type of model you want to run, and `MODEL_PATH` is the path to the model weights or model identifier. For example, to run inference on gpt5.2, you can use the following command:

```bash
python inference.py \
	--model_type gpt-5.2 \
	--model_path gpt-5.2 \
	--split both \
	--output_root OUTPUT_ROOT
```

## Evaluation :straight_ruler:
We use GPT-5-mini as the evaluator to assess model outputs. After running inference, you can evaluate the results with the following command:

```bash
python -m evaluation.eval \
  --inference_root INFERENCE_ROOT \
  --model_type MODEL_TYPE \
  --output_root OUTPUT_ROOT \
  --caption_path CAPTION_PATH
```

Here, `INFERENCE_ROOT` is the path to the inference results, `MODEL_TYPE` is the type of model you evaluated, `output_root` is the location where you want to save your final result,
and `CAPTION_PATH` is the path to the caption file.

## Citing Our Work :star2:

If you find our codebase and dataset beneficial, please cite our work:
```
@article{liu2026mtmcs,
  title={MTMCS-Bench: Evaluating Contextual Safety of Multimodal Large Language Models in Multi-Turn Dialogues},
  author={Liu, Zheyuan and Kim, Dongwhi and Wan, Yixin and Yuan, Xiangchi and Tan, Zhaoxuan and Mo, Fengran and Jiang, Meng},
  journal={arXiv preprint arXiv:2601.06757},
  year={2026}
}
```
# MTMCS-Bench 

Here, we describe how to run models on the MTMCS-Bench benchmark. Due to upload size limits, we do not include the base images or the full set of test samples here. We sincerely apologize for the inconvenience; 
please refer to Appendix I of our paper for sample data. Below, we demonstrate the inference and evaluation workflow for MTMCS-Bench.
## Setup
First, you can install the required packages using requirements.txt

```bash
pip install -r requirements.txt
```
Then, to use the OpenAI and Claude APIs, you need to set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export CLAUDE_API_KEY="your_claude_api_key"
``` 

We currently support 15 models in total; you can find the full list of supported models in the `model` folder. To run inference with a model, use the following command:

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

## Evaluation
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


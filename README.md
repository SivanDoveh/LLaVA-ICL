# 🌋 LLaVA-ICL
## relies heavily on LLaVA. start by having working LLaVa code and enviroment.

Link to files for LLava-ICL [https://drive.google.com/drive/folders/1Zb3sqQaD23gOc0flHqeBmGdFv0T_55kw?usp=sharing]

training_data_mix folder contains: training data (mix of multiple choices, question answering and captioning tasks built from VL checklist and SEED Bench(1-4)

FS_dataset folder contains json files for FS tasks built on 5 known datasets in a format that ICL_model_vqa_FS.py knows to process for evaluation

Model folder contains trained LLAVA-ICL Model that can be evaluated just like vanilla-llava but also on img+text sequences

1. To evaluate a LLaVa model on a single FS model you can use this:
```bash
python llava/eval/ICL_model_vqa_FS.py --question_prompt '{question_prompts}' --episodes_path {path to FS single dataset (CUB/flowers/cars/...)} --model-path {model_path} --output_file 'output_file_name.json'
```
2. During the research process we found out there is a huge performance gap in models when the order of the answers is changed. so we evaluate in both orders with "--reverse_order" and average the results
```bash
python llava/eval/ICL_model_vqa_FS.py --reverse_order --question_prompt '{question_prompts}' --episodes_path {path to FS single dataset (CUB/flowers/cars/...)} --model-path {model_path} --output_file 'output_file_name.json'
```
                    

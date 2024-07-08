# :heart_eyes_cat: LLaVA-ICL WIP!
## Relies heavily on LLaVA. Start by having working LLaVA code and environment.

Link to files for LLava-ICL [https://drive.google.com/drive/folders/1Zb3sqQaD23gOc0flHqeBmGdFv0T_55kw?usp=sharing]

training_data_mix folder contains training data (mix of multiple choices, question answering, and captioning tasks built from VL checklist and SEED Bench(1-4)

FS_dataset folder contains JSON files for FS tasks built on 5 known datasets in a format that ICL_model_vqa_FS.py knows to process for evaluation.

The model folder contains a trained LLAVA-ICL Model that can be evaluated just like vanilla-lava but also on img+text sequences.

1. To evaluate our LLaVa-ICL model on a single FS data(episode path= the path for that FS JSON you should have downloaded from the drive), you can use this line:
```bash
python llava/eval/ICL_model_vqa_FS.py --question_prompt '{question_prompts}' --episodes_path {path to FS single dataset (CUB/flowers/cars/...)} --model-path {model_path} --output_file 'output_file_name.json'
```
2. During the research process we found a huge performance gap in models when the order of the answers is changed. so we evaluate both orders with and without "--reverse_order" and average the results
```bash
python llava/eval/ICL_model_vqa_FS.py --reverse_order --question_prompt '{question_prompts}' --episodes_path {path to FS single dataset (CUB/flowers/cars/...)} --model-path {model_path} --output_file 'output_file_name.json'
```
# questions prompt used for FS evaluation:
```bash
question_prompts=["What is the breed of the dog in the image?","What is the type of the bird in the image?","What is the type of the flower in the image?","What is the type of the food in the image?","What is the model of the car in the image?"]
```
# How does the prompt need to look before getting into LLaVA-ICL?
Use that prompt style, along with your corresponding images. Insert the images as a list of 3 images.
```bash
chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>
What is the type of the flower in the image?
A. pink-yellow dahlia
B. balloon flower
 Answer with the option's letter from the given choices directly. ASSISTANT: A</s>USER: <image>
What is the type of the flower in the image?
A. pink-yellow dahlia
B. balloon flower
 Answer with the option's letter from the given choices directly. ASSISTANT: B</s>USER: <image>
What is the type of the flower in the image?
A. pink-yellow dahlia
B. balloon flower
 Answer with the option's letter from the given choices directly. ASSISTANT:
```

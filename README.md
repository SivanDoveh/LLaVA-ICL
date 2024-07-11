# :heart_eyes_cat: LLaVA-ICL WIP!
## Relies heavily on LLaVA. Start by having a working LLaVA environment.

First, clone this repository:

```bash
git clone git@github.com:SivanDoveh/LLaVA.git
cd LLAVA-ICL
```
Link to files for LLava-ICL [https://drive.google.com/drive/folders/1Zb3sqQaD23gOc0flHqeBmGdFv0T_55kw?usp=sharing]

## Data Preparation
```bash
LLaVA-ICL
├── ALL LLaVA files and folders
│   ├── ...
├── FS_pkls
│   ├── CUB_2way_1shot_episodes.pkl
│   ├── ...
├── data
│   ├── CUB
│   │   ├── CUB_200_2011
│   │   │   ├── images
│   │   │   │   ├── 17.Clay_colored_Sparrow
│   │   │   │   ├── ...
│   ├── flowers
│   │   ├── jpg
│   ├── stanford_dogs
│   │   ├── Images
│   │   │   ├── n02097298-Scotch_terrier
│   │   │   ├── ...
│   ├── food_101
│   │   ├── images
│   │   │   ├── caesar_salad
│   │   │   ├── ...
│   ├── stanford_cars
│   │   ├── images
│   ├── ...
├── ...
```

- Link to files for LLava-ICL [https://drive.google.com/drive/folders/1Zb3sqQaD23gOc0flHqeBmGdFv0T_55kw?usp=sharing]
  - training_data_mix folder contains training data (mix of multiple choices, question answering, and captioning tasks built from VL checklist and SEED Bench(1-4)
  - FS_pkls folder contains pkl files for *TEST* Few Shot tasks built on 5 datasets (food 101, flowers, CUB, Stanford dogs, and Stanford cars) in a format that ICL_model_vqa_FS.py knows to process for evaluation.
  - The model folder contains a trained LLAVA-ICL Model that can be evaluated just like vanilla-lava but also on img+text sequences.

## Few Shot Classification Evaluations on our FS-ICL data
1. To evaluate our LLaVa-ICL model on a single FS data(episode path= the path for that FS JSON you should have downloaded from the drive), you can use this line:
```bash
python llava/eval/ICL_model_vqa_FS.py --question_prompt '{question_prompts}' \
--episodes_path {path to FS single dataset (CUB/flowers/cars/...)} \
--model-path {model_path} --output_file 'output_file_name.json'
```
2. During the research process, we found a huge performance gap in models when the order of the answers changed. so we evaluate both orders with and without "--reverse_order" and average the results
```bash
python llava/eval/ICL_model_vqa_FS.py --reverse_order --question_prompt '{question_prompts}' \
--episodes_path {path to FS single dataset (CUB/flowers/cars/...)} \
--model-path {model_path} --output_file 'output_file_name.json'
```

EXAMPLE for running evaluation on our FS-ICL CUB dataset:
```bash
python llava/eval/ICL_model_vqa_FS.py --question_prompt 'What is the type of the bird in the image?' \
--episodes_path './FS_pkls/CUB_2way_1shot_episodes.pkl' \
--model-path path/to/model/folder/train_llava_icl_mix_llava_seed_Vl_ALL_QA_MC_NEW_Cap --output_file 'out.json'
```

#### questions prompt used for FS-ICL classification evaluation:
```bash
question_prompts=["What is the breed of the dog in the image?","What is the type of the bird in the image?" \
,"What is the type of the flower in the image?","What is the type of the food in the image?", \
"What is the model of the car in the image?"]
```
## Few Shot ICL Classification Evaluations on YOUR data
you need to prepare a list of dictionaries in this format

{'test_image': 'path/to/image/query_image.jpg', 'test_class': 'class of test image- same as positive example class', 'positive_images': ['path/to/positive class image'], 'negs': [{'neg_images': ['path/to/negative class image'], 'neg_class': 'class of negative image'}]}]

#### How does the prompt need to look before getting into LLaVA-ICL?
The FS dataset and the data loader in ICL_model_vqa_FS will convert your pickle data file to look like this conversation: (Insert the images as a list of 3 images)

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

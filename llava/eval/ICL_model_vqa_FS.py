import argparse
import copy

import torch
import os
import json
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.eval.FS_dataset import get_dataloader
from tqdm import tqdm


def format_sentences(sentences,args):
    formatted_sentence = ''
    list_of_chars=[]
    # Starting from the ASCII value of 'A'
    char_code = ord('A')
    if args.reverse_order:
        sentences.reverse()
    for sentence in sentences:
        # Prepending the character and a period
        formatted_sentence += f"{chr(char_code)}. {sentence}\n"
        # Move to the next character
        list_of_chars.append(f"{chr(char_code)}")
        char_code += 1
    if args.reverse_order:
        list_of_chars.reverse()
    return formatted_sentence,list_of_chars

def eval_model(args):
    # Model
    list_conv =[]
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    dataloader = get_dataloader(args,image_processor)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    counter=0
    for i, data in enumerate(tqdm(dataloader)): 
        images_shot_tensors,desc_shots_list,images_test_tensors,desc_test_list,images_pathes = data
        # create in context (<img> + text
        conv = conv_templates[args.conv_mode].copy()
        qs1 = DEFAULT_IMAGE_TOKEN + '\n'
        clean_desc_shots_list=[]

        question=f'{args.question_prompt}'+ '\n'
        for desc in desc_shots_list:
            clean_desc_shots_list.append(desc[0].strip('\'"'))
        for ind,desc in enumerate(desc_shots_list):
            turn=qs1
            turn += question
            answers, list_of_chars = format_sentences(copy.deepcopy(clean_desc_shots_list),args)
            turn += f"{answers} Answer with the option's letter from the given choices directly."
            conv.append_message(conv.roles[0], turn)
            conv.append_message(conv.roles[1], list_of_chars[ind])
        #test image
        turn=qs1
        turn += question
        answers, list_of_chars = format_sentences(copy.deepcopy(clean_desc_shots_list), args)
        turn += f"{answers} Answer with the option's letter from the given choices directly."
        conv.append_message(conv.roles[0], turn)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        for test_ind in range(images_test_tensors.shape[1]):
            images_tensors = torch.cat([images_shot_tensors, images_test_tensors[:,test_ind].unsqueeze(dim=1)], dim=1)[0]
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensors.half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            GT = list_of_chars[test_ind]
            counter += GT==outputs

            if (i + 1) % 100 == 0:
                print(f"i={i} accuracy: {str(counter/(i+1))}")

    print(f"accuracy: {str(counter/(i+1))}")
    item = {
        "acc": counter/(i+1),
    }
    with open(args.output_file, "w") as f:
        json.dump(item, f, indent=2)

    return counter/(i+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='path/to/model')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--curr_chunk", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)#0.2
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--reverse_order", default=False, action="store_true")
    parser.add_argument("--output_file", type=str, default="Output_file.json")
    parser.add_argument("--episodes_path", type=str, default="path/to/fs/data/flowers_dataset_2way_1shot_episodes.pkl")
    parser.add_argument("--question_prompt", type=str, default="What is the type of the flower in the image?")
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--bs", type=int, default=1)
    args = parser.parse_args()

    acc_reg = eval_model(args)
    args.reverse_order = True
    acc_rev = eval_model(args)
    print(f"{args.episodes_path} accuracy: {str((acc_reg+acc_rev)/2)}")


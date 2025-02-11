import os
import json
import random
import numpy as np
import copy
import transformers
from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer, AutoModelForCausalLM

class PMC_Alignment_Dataset(Dataset):
    def __init__(self, img_dir, json_path, tokenizer_path, img_tokens=32, seq_length=512, mode='Train', start=0, text_type='random', no_image=False):
        self.img_root = img_dir
        
        # Load JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)[start:]
        
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1
        
        self.img_padding = [-100 for _ in range(img_tokens)]
        self.attn_padding = [1 for _ in range(img_tokens)]
        
        self.H = 512
        self.W = 512
        self.C = 3
        self.text_type = text_type
        self.no_image = no_image

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((self.H, self.W), scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        if mode == 'Test':
            self.transform = transforms.Compose([
                transforms.Resize((self.H, self.W), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

            if self.text_type == 'random':
                self.text_type = 'caption'

        self.mode = mode
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def random_answer(self, question, gpt_response):
        p = random.random()

        if self.text_type == 'random':
            if p <= 0.5:
                pre_text = question
                final_text = question+gpt_response
            else:
                pre_text = ''
                final_text = gpt_response
        elif self.text_type == 'caption':
            pre_text = question
            final_text = question+gpt_response
        else:
            pre_text = ''
            final_text = question+gpt_response

        return pre_text, final_text

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample['conversatons'][0]['value'].replace('\n<image>', '')
        gpt_response = sample['conversatons'][1]['value']

        if not self.no_image:
            img_path = os.path.join(self.img_root, sample['image'])
            img = PIL.Image.open(img_path).convert('RGB')
            image = self.transform(img)

        if self.mode == 'Train':
            pre_text, final_o = self.random_answer(question, gpt_response)

            final_o = self.tokenizer(final_o)
            input_ids = final_o['input_ids']
            input_ids.append(self.tokenizer.eos_token_id)
            input_ids = np.array(input_ids)

            if len(input_ids) < self.seq_length:
                input_ids = np.pad(input_ids, (0, self.seq_length - len(input_ids)), 'constant', constant_values=0)
            else:
                input_ids = input_ids[:self.seq_length]
                input_ids[-1] = self.tokenizer.eos_token_id

            label = copy.deepcopy(input_ids)
            label[label == 0] = -100

            if pre_text != '':
                pre_text = self.tokenizer(pre_text)
                if len(pre_text['input_ids']) < len(label):
                    label[:len(pre_text['input_ids'])] = -100

            label = label.tolist()

            if not self.no_image:
                label = np.array(self.img_padding + label)

                item = {
                    'input_ids': input_ids,
                    'images': image,
                    'labels': label,
                }
            else:
                label = np.array(label)
                item = {
                    'input_ids': input_ids,
                    'labels': label,
                }

            return item

        if self.mode == 'Test':
            if not self.no_image:
                item = {
                    'input_ids': question,
                    'img_path': sample['image'],
                    'images': image,
                    'labels': gpt_response,
                }
            else:
                item = {
                    'input_ids': question,
                    'img_path': sample['image'],
                    'labels': gpt_response,
                }

            return item

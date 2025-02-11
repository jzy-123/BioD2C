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
from .randaugment import RandomAugment    
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

class BioVGQ(Dataset):
    def __init__(self, img_dir, jsonl_path, tokenizer_path, img_tokens=32, seq_length=512, mode='Train', start=0, no_image=False):
        self.img_root = img_dir
        
        # Load JSON data
        with open(jsonl_path, 'r') as f:
            self.data = [json.loads(line) for line in f][start:]
        
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,token=access_token)
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1

        self.img_tokens = img_tokens
        
        self.img_padding = [-100 for _ in range(img_tokens)]
        self.attn_padding = [1 for _ in range(img_tokens)]
        
        self.H = 512
        self.W = 512
        self.C = 3
        self.no_image = no_image

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop((self.H, self.W), scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ]) 

        if mode == 'Test':
            self.transform = transforms.Compose([
                transforms.Resize((self.H, self.W), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        self.mode = mode
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        sample = self.data[idx]
        question = str(sample["question"])
        answer = str(sample["answer"])

        pre_text = "Question: "+question+" Answer:"
        final_text = "Question: "+question+" Answer:"+answer

        if not self.no_image:
            img_path = os.path.join(self.img_root, sample['image'])
            img = PIL.Image.open(img_path).convert('RGB')
            image = self.transform(img)

        queue_text = None
        queue_length = None
        if 'queue' in sample and sample["queue"] is not None and self.mode == 'Train':
            queue_text = sample['queue'].split(",")
            queue_text = [self.tokenizer(text, padding='max_length', truncation=True, max_length=self.img_tokens)['input_ids'] for text in queue_text]
            queue_text = [np.array(text) for text in queue_text]
            if len(queue_text) < 10:
                queue_text.extend([np.zeros_like(queue_text[0]) for _ in range(10 - len(queue_text))])
            else:
                queue_text = queue_text[:10]
            queue_text = np.array(queue_text)


        if self.mode == 'Train':
            pre_text, final_o = self.random_answer(question, answer)

            final_o = self.tokenizer(final_o)
            input_ids = final_o['input_ids']
            input_ids.append(self.tokenizer.eos_token_id)
            input_ids = np.array(input_ids)
            
            if len(input_ids) < self.seq_length:
                input_ids = np.pad(input_ids, (0, self.seq_length - len(input_ids)), 'constant', constant_values=self.tokenizer.pad_token_id)
            else:
                input_ids = input_ids[:self.seq_length]
                input_ids[-1] = self.tokenizer.eos_token_id
                
            label = copy.deepcopy(input_ids)
            label[label == self.tokenizer.pad_token_id] = -100
            if pre_text != '':
                pre_text = self.tokenizer(pre_text)['input_ids']
                if len(pre_text) < len(label):
                    label[:len(pre_text)] = -100
            else:
                pre_text = None
            label = label.tolist()
            if len(pre_text)<self.img_tokens:
                pre_text = np.pad(pre_text,(0, self.img_tokens - len(pre_text)), 'constant', constant_values=0)
            else:
                pre_text = pre_text[:self.img_tokens]

            if not self.no_image:
                label = np.array(self.img_padding + label)
                item = {
                    'input_ids': input_ids,       
                    'images': image,
                    'labels': label,
                    'queue_text': queue_text,
                    'pre_text':np.array(pre_text),
                }
            else:
                label = np.array(label)
                item = {
                    'input_ids': input_ids,   
                    'labels': label,
                    'queue_text': queue_text,
                    'pre_text':np.array(pre_text),
                }
            return item

        if self.mode == 'Test':
            if not self.no_image:
                item = {
                    'input_ids': "Question: "+question+" Answer:",
                    'img_path': sample['image'],
                    'images': image,
                    'labels': answer,
                }
            else:
                item = {
                    'input_ids': "Question: "+question+" Answer:",
                    'img_path': sample['image'],
                    'labels': answer,
                }

            return item

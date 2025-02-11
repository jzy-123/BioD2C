import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import PIL
import numpy as np
import torch.nn.functional as F
import transformers
import pandas as pd
import random
import copy
from .randaugment import RandomAugment    
from PIL import Image
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
    
class Slake_Dataset(Dataset):
    def __init__(self,  csv_path, tokenizer_path, img_dir = '',img_tokens = 32, seq_length = 512,voc_size = 32000, mode = 'Train',start = 0):
        self.img_root = img_dir
        self.data = pd.read_csv(csv_path).iloc[start:]

        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1

        self.mode = mode
        self.img_padding = [-100 for i in range(img_tokens)]
        self.attn_padding = [1 for i in range(img_tokens)]
        self.H = 512
        self.W = 512
        self.C = 3
        self.img_tokens = img_tokens
        
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop((self.H,self.W),scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                #transforms.RandomHorizontalFlip(),
                RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',]),     
                transforms.ToTensor(),
                normalize,
            ]) 
        if self.mode == 'Test':
            self.transform = transforms.Compose([                        
                    transforms.Resize((self.H,self.W), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    normalize,
                ])
            
        self.mode = mode
        self.seq_length = seq_length
        self.voc_size = voc_size
    
    def random_answer(self, Question,Answer):
        Answer = str(Answer)
        pre_text = 'Question: '+ Question +' Answer:'
        final_o = 'Question: '+ Question +' Answer:' + Answer
        return pre_text,final_o
     
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        Question  = str(sample['question'])
        Anwser = str(sample['answer'])
        
        ##### read image pathes #####
        img_path = self.img_root + sample['img_name']
        img = PIL.Image.open(img_path).convert('RGB') 
        image = self.transform(img) 
        
        if self.mode == 'Train':
            pre_text,final_o = self.random_answer(Question,Anwser)
            
            final_o = self.tokenizer(final_o)
            input_ids = final_o['input_ids']
            input_ids.append(self.tokenizer.eos_token_id)
            input_ids = np.array(input_ids)
            
            if len(input_ids) < self.seq_length:
                input_ids = np.pad(input_ids, (0, self.seq_length - len(input_ids)), 'constant', constant_values=self.tokenizer.pad_token_id)
            else:
                input_ids = input_ids[:self.seq_length]
                
            #attention = np.array(self.attn_padding + final_o['attention_mask'])
            label = copy.deepcopy(input_ids)
            label[label==self.tokenizer.pad_token_id] = -100
            if pre_text != '':
                pre_text = self.tokenizer(pre_text)['input_ids']
                if len(pre_text)<len(label):
                    #label = np.array(label)
                    label[:len(pre_text)] = -100
            label = label.tolist()
            label = np.array(self.img_padding + label)
            if len(pre_text)<self.img_tokens:
                pre_text = np.pad(pre_text,(0, self.img_tokens - len(pre_text)), 'constant', constant_values=0)
            else:
                pre_text = pre_text[:self.img_tokens]
            
            item = {
                'input_ids': input_ids,       
                'images': image,
                'labels': label,
                'pre_text':pre_text
            }
                
        if self.mode == 'Test':
            item = {
                'input_ids': 'Question: '+ Question + ' Answer:' ,
                'img_path': sample['img_name'],       
                'images': image,
                'labels': Anwser,
            }
        return item
        


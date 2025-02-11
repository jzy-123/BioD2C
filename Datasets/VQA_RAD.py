import torch
from torch.utils.data import Dataset
from torchvision import transforms
import xml.etree.ElementTree as ET
import PIL
import numpy as np
import copy

import transformers
from .randaugment import RandomAugment
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM


class VQA_RAD_Dataset(Dataset):
    def __init__(self, xml_path, tokenizer_path, img_dir, img_tokens=32, seq_length=512, voc_size=32000, mode='Train', start=0):
        self.img_root = img_dir
        self.data = self._parse_xml(xml_path)[start:]

        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1
        self.mode = mode
        self.img_padding = [-100 for _ in range(img_tokens)]
        self.attn_padding = [1 for _ in range(img_tokens)]
        self.H = 512
        self.W = 512
        self.C = 3

        self.img_tokens = img_tokens

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((self.H, self.W), scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
            transforms.ToTensor(),
            normalize,
        ])
        if self.mode == 'Test':
            self.transform = transforms.Compose([
                transforms.Resize((self.H, self.W), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        self.seq_length = seq_length
        self.voc_size = voc_size

    def __len__(self):
        return len(self.data)

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        data = []
        for question in root.findall('question'):
            item = {
                'question': question.find('question').text,
                'answer': question.find('answer').text,
                'img_name': question.find('image_name').text
            }
            data.append(item)
        return data

    def random_answer(self, Question, Answer):
        Answer = str(Answer)
        pre_text = "Question: "+Question+" Answer:"
        final_text = "Question: "+Question+" Answer:"+Answer
        return pre_text, final_text

    def __getitem__(self, idx):
        sample = self.data[idx]
        Question = str(sample['question'])
        Answer = str(sample['answer'])

        # Read image
        img_path = self.img_root + sample['img_name']
        img = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(img)

        if self.mode == 'Train':
            pre_text, final_o = self.random_answer(Question, Answer)

            final_o = self.tokenizer(final_o)
            input_ids = final_o['input_ids']
            input_ids.append(self.tokenizer.eos_token_id)
            input_ids = np.array(input_ids)

            if len(input_ids) < self.seq_length:
                input_ids = np.pad(input_ids, (0, self.seq_length - len(input_ids)), 'constant', constant_values=0)
            else:
                input_ids = input_ids[:self.seq_length]

            label = copy.deepcopy(input_ids)
            label[label == 0] = -100
            if pre_text != '':
                pre_text = self.tokenizer(pre_text)['input_ids']
                if len(pre_text) < len(label):
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
                'input_ids': 'Question: ' + Question + ' Answer:',
                'img_path': sample['img_name'],
                'images': image,
                'labels': Answer,
            }

        return item

        

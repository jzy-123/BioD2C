import safetensors
import tqdm.auto as tqdm
from typing import Optional
import transformers
from Datasets.VQA_RAD import VQA_RAD_Dataset
from models.model import QA_model
from dataclasses import dataclass, field
from torch.utils.data import DataLoader  
import torch
import csv
@dataclass
class ModelArguments:
    train_stage:Optional[int] = field(default=2)
    use_queue: Optional[bool] = field(default=False)
    model_path: Optional[str] = field(default="chaoyi-wu/PMC_LLAMA_7B")
    ckp: Optional[str] = field(default="")
    checkpointing: Optional[bool] = field(default=False)
    ## Q_former ##
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=8)
    img_token_num: Optional[int] = field(default=32)
    
    ## Basic Setting ##
    voc_size: Optional[int] = field(default=128256)
    hidden_dim: Optional[int] = field(default=4096)
    
    ## Image Encoder ##
    Vision_module: Optional[str] = field(default='PMC-CLIP')
    visual_model_path: Optional[str] = field(default='./models/pmc_clip/checkpoint.pt')
    
    ## Peft ##
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)

@dataclass
class DataArguments:
    img_dir: str = field(default='./Data/VQA_RAD/VQA_RAD Image Folder/', metadata={"help": "Path to the training data."})
    Test_data_path: str = field(default='./Data/VQA_RAD/test_close.xml', metadata={"help": "Path to the training data."})
    tokenizer_path: str = field(default='chaoyi-wu/PMC_LLAMA_7B', metadata={"help": "Path to the training data."})
    trier: int = field(default=0)
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./Results")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


  
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Data")
    row_count = 0
    
    Test_dataset_close = VQA_RAD_Dataset(data_args.Test_data_path, data_args.tokenizer_path,img_dir=data_args.img_dir,mode='Test',start=row_count)
    
    # batch size should be 1
    Test_dataloader_close = DataLoader(
            Test_dataset_close,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
    ) 
    Test_dataset_open = VQA_RAD_Dataset(data_args.Test_data_path.replace('close.xml','open.xml'), data_args.tokenizer_path,img_dir=data_args.img_dir,mode='Test',text_type='blank',start=row_count)
    
    # batch size should be 1
    Test_dataloader_open = DataLoader(
            Test_dataset_open,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
    )  

    print("Setup Model")
    ckp = model_args.ckp + '/model.safetensors'
    print(ckp)
    ckpt = safetensors.torch.load_file(ckp)

    model = QA_model(model_args)

    model.load_state_dict(ckpt)

    model.eval()

    
    with open('./rescsv/VQA_RAD/RAD_close_'+model_args.ckp.split('/')[-3]+'_'+ model_args.ckp.split('/')[-2]+'.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Figure_path','Question','Pred','Label'])
            for sample in tqdm.tqdm(Test_dataloader_close):
                input_ids = Test_dataset_close.tokenizer(sample['input_ids'],return_tensors="pt")
                #input_ids['input_ids'][0][0]=1
                images = sample['images']
                with torch.no_grad():
                    generation_ids = model.generate_long_sentence(input_ids['input_ids'],images)
                generated_texts = Test_dataset_close.tokenizer.batch_decode(generation_ids, skip_special_tokens=True) 
                for i in range(len(generated_texts)):
                    label = sample['labels'][i]
                    img_path = sample['img_path'][i]
                    pred = generated_texts[i]
                    writer.writerow([img_path,sample['input_ids'][i],pred,label])
    
    with open('./rescsv/VQA_RAD/RAD_open_'+model_args.ckp.split('/')[-3]+'_'+ model_args.ckp.split('/')[-2]+'.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Figure_path','Question','Pred','Label'])
            for sample in tqdm.tqdm(Test_dataloader_open):
                input_ids = Test_dataset_open.tokenizer(sample['input_ids'],return_tensors="pt")
                #input_ids['input_ids'][0][0]=1
                images = sample['images']
                with torch.no_grad():
                    generation_ids = model.generate_long_sentence(input_ids['input_ids'],images)
                generated_texts = Test_dataset_open.tokenizer.batch_decode(generation_ids, skip_special_tokens=True) 
                for i in range(len(generated_texts)):
                    label = sample['labels'][i]
                    img_path = sample['img_path'][i]
                    pred = generated_texts[i]
                    writer.writerow([img_path,sample['input_ids'][i],pred,label])
             
if __name__ == "__main__":
    main()


import tqdm.auto as tqdm
from typing import Optional
import transformers
from Datasets.PMC_600k import PMC_Alignment_Dataset
from models.model import QA_model
from transformers import Trainer
from dataclasses import dataclass, field 

class VQATrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model
        if self.optimizer is None:
            # 1. Freeze all parameters initially
            for param in  opt_model.llamacasual.parameters():
                param.requires_grad = False
            
            for param in opt_model.vision_model.parameters():
                param.requires_grad = False

            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer


@dataclass
class ModelArguments:
    use_queue:Optional[bool] = field(default=False)
    train_stage: Optional[int] = field(default=1)
    model_path: Optional[str] = field(default="chaoyi-wu/PMC_LLAMA_7B")
    ckp: Optional[str] = field(default="")
    ## Q_former ##
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=8)
    img_token_num: Optional[int] = field(default=32)
    
    ## Basic Setting ##
    hidden_dim: Optional[int] = field(default=4096)
    checkpointing: Optional[bool] = field(default=True)
    ## Image Encoder ##
    Vision_module: Optional[str] = field(default='PMC-CLIP')
    visual_model_path: Optional[str] = field(default='./models/pmc_clip/checkpoint.pt')
    ## Peft ##
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)

@dataclass
class DataArguments:
    img_dir: str = field(default='./Data/PMC-600K/images', metadata={"help": "Path to the training data."})
    Train_data_path: str = field(default='', metadata={"help": "Path to the training data."})
    Eval_data_path: str = field(default='', metadata={"help": "Path to the training data."})
    tokenizer_path: str = field(default='chaoyi-wu/PMC_LLAMA_7B', metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: Optional[bool] = field(default=False)

        
    
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Data")
    Train_dataset = PMC_Alignment_Dataset(data_args.img_dir, data_args.Train_data_path, data_args.tokenizer_path, text_type = 'caption')
    Eval_dataset = PMC_Alignment_Dataset(data_args.img_dir, data_args.Eval_data_path, data_args.tokenizer_path, text_type = 'caption')

    print("Setup Model")
    model = QA_model(model_args)
    run_name_root = training_args.run_name
    output_dir_root = training_args.output_dir
    
    training_args.run_name = run_name_root+'_stage1'
    training_args.output_dir = output_dir_root + '/stage1/'
    
    print('Start Pretraining')
    trainer = VQATrainer(model=model, 
                      train_dataset = Train_dataset, 
                      eval_dataset = Eval_dataset,
                      args=training_args,
                      )
    try:
        trainer.train(resume_from_checkpoint=True)
    except:
        trainer.train()
    trainer.save_state()

    
if __name__ == "__main__":
    main()
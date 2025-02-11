import numpy as np
from .transformer import *
from dataclasses import dataclass, field
import tqdm.auto as tqdm

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from einops import rearrange
import transformers
from transformers import CLIPVisionConfig
from .blocks import ModifiedResNet,PMC_CLIP_cfg
import torchvision.models as models
import json
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)



@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    model_path: str = field()


@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    num_virtual_tokens: int = field(default=32)  # Used for prompt tuning, prefix tuning and p-tuning
    mapping_hidden_dim: int = field(default=1024)


def get_peft_config(peft_args: PEFTArguments):
    if peft_args.peft_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=peft_args.lora_rank,
            lora_alpha=32, lora_dropout=0.1
        )
    elif peft_args.peft_mode == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
            prefix_projection=True,
        )
    elif peft_args.peft_mode == "ptuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
        )
    elif peft_args.peft_mode == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
        )
    else:
        raise KeyError(peft_args.peft_mode)
    return peft_config

class Text_Image_MLP(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=2048, output_dim=1024):
        super(Text_Image_MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()  # 激活函数: ReLU
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x) 
        x = self.relu(x) 
        x = self.layer2(x) 
        return x
    

  
class QA_model(nn.Module):
    def __init__(self, model_args):  
        super(QA_model, self).__init__()  
        self.use_queue = model_args.use_queue
        self.train_stage = model_args.train_stage
        self.hidden_dim = model_args.hidden_dim
        self.voc_size = model_args.voc_size
        self.img_tokens = model_args.img_token_num
        self.H = model_args.H 
        self.N = model_args.N 
        self.Vision_module = model_args.Vision_module
        


        vision_cfg = PMC_CLIP_cfg()
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        vision_model = ModifiedResNet(
                layers=vision_cfg.layers,
                heads=vision_heads,
                output_dim = 768,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width
            )
        vision_model = self.vision_load_pretrain(vision_model,model_args.visual_model_path)
        self.vision_model = nn.Sequential(*list(vision_model.children())[:-2])
        num_ftrs = 1024
    
        
        self.fc_hpm = nn.Linear(num_ftrs,num_ftrs)
        
        self.mlp = Text_Image_MLP(input_dim=self.hidden_dim,output_dim=num_ftrs)

        self.query_embed1 = nn.Embedding(self.img_tokens, num_ftrs) 
        self.query_embed2 = nn.Embedding(self.img_tokens, num_ftrs)
        
        decoder_layer = TransformerDecoderLayer(num_ftrs, self.H, 1024,
                                        0.1, 'relu',normalize_before=True)
        decoder_norm = nn.LayerNorm(num_ftrs)
        self.decoder = TransformerDecoder(decoder_layer, self.N , decoder_norm,
                                  return_intermediate=False)
        
        decoder_layer_hpm = TransformerDecoderLayer(num_ftrs, self.H, 1024,
                                        0.1, 'relu',normalize_before=True)
        decoder_norm_hpm = nn.LayerNorm(num_ftrs)
        self.decoder_hpm = TransformerDecoder(decoder_layer_hpm, self.N , decoder_norm_hpm,
                                  return_intermediate=False)
     
        self.fc_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.fc_l2 = nn.Linear(num_ftrs, self.hidden_dim)

        self.beta = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.gate_activation = nn.Tanh()
        self.fc_l1_gate = nn.Linear(num_ftrs, num_ftrs)
        self.fc_l2_gate = nn.Linear(num_ftrs, self.hidden_dim)
    
        self.llamacasual = self.Setup_model(model_args)
        
    def vision_load_pretrain(self,resnet,model_path):
        checkpoint = torch.load(model_path, map_location='cpu') 
        state_dict = checkpoint['state_dict'] 
        state_dict = {k.replace('module.visual.',''): v for k, v in state_dict.items() if '.visual' in k}
        resnet.load_state_dict(state_dict)
        return resnet
                
    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")    

    def Setup_model(self, model_args):
        print("Setup Model")
        model = transformers.LlamaForCausalLM.from_pretrained(
           model_args.model_path,
        )
        if model_args.checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = False
        if model_args.is_lora:
            print("Setup PEFT")
            peft_config = get_peft_config(peft_args=model_args)
            model = get_peft_model(model, peft_config)
        return model
    
    def split_and_pool(self,image_feature_map, s):
        B, H_prime, W_prime, D = image_feature_map.shape
        assert H_prime == W_prime, "The input feature map must be square!"
        
        N = sum(2**i for i in range(1, s + 1))
        pooled_vectors = []

        for level in range(1, s + 1):
            num_patches = 2 ** (level - 1)
            patch_size = H_prime // num_patches
            
            for i in range(num_patches):
                for j in range(num_patches):
                    start_h = i * patch_size
                    end_h = (i + 1) * patch_size
                    start_w = j * patch_size
                    end_w = (j + 1) * patch_size
                    patch = image_feature_map[:, start_h:end_h, start_w:end_w, :]  # (B, patch_size, patch_size, D)

                    gap = torch.mean(patch, dim=(1, 2))
                    gmp = torch.max(patch.reshape(B, -1, D), dim=1)[0]
                    
                    pooled_vector = gap + gmp  # (B, D)
                    pooled_vectors.append(pooled_vector.unsqueeze(1))

        pooled_vectors = torch.cat(pooled_vectors, dim=1) #(B, M, D)
        return pooled_vectors
    
    def image_encoder(self, xis):
        batch_size = xis.shape[0]
        res_fea = self.vision_model(xis) #batch_size,feature_size,patch_num,patch_num
        out_emb = rearrange(res_fea,'b d n1 n2 -> b n1 n2 d')
        hpm = self.split_and_pool(out_emb,s=6)
        out_emb = rearrange(res_fea,'b d n1 n2 -> b (n1 n2) d')
        return out_emb,hpm
    
    def calculate_cosine_similarity_with_queue(self, text_features, image_features, queue_embeddings, tau=0.5):
        b, l, n, d = queue_embeddings.shape

        text_features = F.normalize(text_features, dim=-1)  # (b, n, d)
        image_features = F.normalize(image_features, dim=-1)  # (b, n, d)
        queue_embeddings = F.normalize(queue_embeddings, dim=-1)  # (b, l, n, d)

        text_features = text_features.unsqueeze(1)  # (b, 1, n, d)
        text_similarities = torch.einsum('bind,blnd->bln', text_features, queue_embeddings)  # (b, l, n)
        text_similarities = text_similarities.mean(dim=-1)

        image_features = image_features.unsqueeze(1)  # (b, 1, n, d)
        image_similarities = torch.einsum('bind,blnd->bln', image_features, queue_embeddings)  # (b, l, n)
        image_similarities = image_similarities.mean(dim=-1)

        text_similarities = text_similarities / tau
        text_similarities = text_similarities - text_similarities.max(dim=-1, keepdim=True)[0]
        text_similarities = F.softmax(text_similarities, dim=-1)  # (b, l)

        image_similarities = image_similarities / tau
        image_similarities = image_similarities - image_similarities.max(dim=-1, keepdim=True)[0]
        image_similarities = F.softmax(image_similarities, dim=-1)  # (b, l)

        return text_similarities, image_similarities

    
    def forward(self,input_ids,images,pre_text=None ,labels=None,queue_text=None):
        
        B = images.shape[0]
        ### images encoding ###
        x,hpm = self.image_encoder(images)
        features = x.transpose(0,1) #patch_num b dim
        f_hpm = hpm.transpose(0,1)
        #text query
        text_emb = self.llamacasual.get_input_embeddings()(pre_text)
        text_query = self.mlp(text_emb).transpose(0,1)
        query = self.query_embed1.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
        
        features,ws = self.decoder(query, features, memory_key_padding_mask=None, pos=None, query_pos=None)
        features = features.transpose(0,1)

        f_hpm,_ = self.decoder(text_query,f_hpm,memory_key_padding_mask=None, pos=None, query_pos=None)
        f_hpm = f_hpm.transpose(0,1)

        ### fc  ### 
        features = rearrange(features,'b n d  -> (b n) d')
        features = self.fc_l1(features)
        features = F.relu(features)
        features = self.fc_l2(features)
        features = rearrange(features,'(b n) d -> b n d',b=B)

        f_hpm = rearrange(f_hpm,'b n d  -> (b n) d')
        f_hpm = self.fc_l1_gate(f_hpm)
        f_hpm = F.relu(f_hpm)
        f_hpm = self.fc_l2_gate(f_hpm)
        f_hpm = rearrange(f_hpm,'(b n) d -> b n d',b=B)

        image_features = self.gate_activation(self.beta)*f_hpm+features

        #queue
        kl_loss = None
        if self.use_queue:
            if queue_text is not None:
                queue_emb = self.llamacasual.get_input_embeddings()(queue_text)
                xt,xv = self.calculate_cosine_similarity_with_queue(text_emb,features,queue_emb,tau=0.1)
                kl_loss=F.kl_div(xv.log(),xt,reduction="batchmean")
                
        ### LLM ###
        input_embedding = self.llamacasual.get_input_embeddings()(input_ids)
        input_embedding = torch.cat([image_features,input_embedding], dim=1)       
        output = self.llamacasual(inputs_embeds = input_embedding, labels = labels)
        if kl_loss is not None:
            output.loss = output.loss + kl_loss
        
        return output

    def generate(self,input_ids,images):
        with torch.no_grad():
            B = images.shape[0]
            ### images encoding ###
            x,hpm = self.image_encoder(images)
            features = x.transpose(0,1) #patch_num b dim
            f_hpm = hpm.transpose(0,1)

            #text query
            if self.train_stage==2:
                text_query_ids = input_ids[:,:self.img_tokens]
                text_emb = self.llamacasual.get_input_embeddings()(text_query_ids)
                query1 = self.mlp(text_emb).transpose(0,1)
                query2 = self.mlp(text_emb).transpose(0,1)
            else:
                query1 = self.query_embed1.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
                query2 = self.query_embed2.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
            
            features,ws = self.decoder(query1, features, memory_key_padding_mask=None, pos=None, query_pos=None)
            features = features.transpose(0,1)

            f_hpm,_ = self.decoder_hpm(query2,f_hpm,memory_key_padding_mask=None, pos=None, query_pos=None)
            f_hpm = f_hpm.transpose(0,1)

            ### fc  ### 
            features = rearrange(features,'b n d  -> (b n) d')
            features = self.fc_l1(features)
            features = F.relu(features)
            features = self.fc_l2(features)
            features = rearrange(features,'(b n) d -> b n d',b=B)

            f_hpm = rearrange(f_hpm,'b n d  -> (b n) d')
            f_hpm = self.fc_l1_gate(f_hpm)
            f_hpm = F.relu(f_hpm)
            f_hpm = self.fc_l2_gate(f_hpm)
            f_hpm = rearrange(f_hpm,'(b n) d -> b n d',b=B)

            image_features = self.gate_activation(self.beta)*f_hpm+features

            ### LLM ###
            input_embedding = self.llamacasual.get_input_embeddings()(input_ids)
            input_embedding = torch.cat([image_features,input_embedding], dim=1)
                
            generation = self.llamacasual(inputs_embeds = input_embedding)['logits']
            #generation = self.llamacasual.generate(inputs_embeds = input_embedding,max_length=100, do_sample=True, top_k=50)
            return generation
    
    def generate_long_sentence(self,input_ids,images):
        with torch.no_grad():
            B = images.shape[0]
            ### images encoding ###
            x,hpm = self.image_encoder(images)
            features = x.transpose(0,1) #patch_num b dim
            f_hpm = hpm.transpose(0,1)

            #text query
            if input_ids.shape[1] > self.img_tokens:
                text_query_ids = input_ids[:, :self.img_tokens]
            else:
                padding = torch.zeros(
                    (input_ids.size(0), self.img_tokens - input_ids.size(1)),
                    dtype=input_ids.dtype,
                    device=input_ids.device
                )
                text_query_ids = torch.cat((input_ids, padding), dim=1)
            text_emb = self.llamacasual.get_input_embeddings()(text_query_ids)
            text_query= self.mlp(text_emb).transpose(0,1)
           
            query = self.query_embed1.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
            
            features,ws = self.decoder(query, features, memory_key_padding_mask=None, pos=None, query_pos=None)
            features = features.transpose(0,1)

            f_hpm,_ = self.decoder(text_query,f_hpm,memory_key_padding_mask=None, pos=None, query_pos=None)
            f_hpm = f_hpm.transpose(0,1)

            ### fc  ### 
            features = rearrange(features,'b n d  -> (b n) d')
            features = self.fc_l1(features)
            features = F.relu(features)
            features = self.fc_l2(features)
            features = rearrange(features,'(b n) d -> b n d',b=B)

            f_hpm = rearrange(f_hpm,'b n d  -> (b n) d')
            f_hpm = self.fc_l1_gate(f_hpm)
            f_hpm = F.relu(f_hpm)
            f_hpm = self.fc_l2_gate(f_hpm)
            f_hpm = rearrange(f_hpm,'(b n) d -> b n d',b=B)

            image_features = self.gate_activation(self.beta)*f_hpm+features
            # print(self.beta)

            ### LLM ###
            input_embedding = self.llamacasual.get_input_embeddings()(input_ids)
            input_embedding = torch.cat([image_features,input_embedding], dim=1)
            
            #generation = self.llamacasual(inputs_embeds = input_embedding)['logits']
            generation = self.llamacasual.generate(inputs_embeds = input_embedding, 
                                                   max_new_tokens=50,
                                                  )
            return generation

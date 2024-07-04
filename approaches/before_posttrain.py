import logging
import math

import numpy as np
import os
import torch
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from utils import utils
from networks.baselines import ewc, hat, softmask, memory, demix


def prepare(self,model, train_loader_subset, train_loader_subset_dataset, accelerator):
    self_fisher = None
    mask_pre = None
    mask_back = None
    buffer = None
    head_impt = None
    intermediate_impt = None
    output_impt = None


    if 'ewc' in self.args.baseline:
        if os.path.exists(os.path.join(self.args.output_dir + '../', 'fisher')):
            print('Load fisher matrix **************')
            self_fisher = torch.load(os.path.join(self.args.output_dir + '../', 'fisher'))
            for k, v in self_fisher.items():
                self_fisher[k] = self_fisher[k].cuda()

    elif 'adapter_hat' in self.args.baseline \
            or 'transformer_hat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:  # BCL included HAT
        if os.path.exists(os.path.join(self.args.output_dir + '../', 'mask_pre')):
            print('Load mask matrix **************')
            mask_pre = torch.load(os.path.join(self.args.output_dir + '../', 'mask_pre'))
            mask_back = torch.load(os.path.join(self.args.output_dir + '../', 'mask_back'))

            for k, v in mask_pre.items():
                mask_pre[k] = mask_pre[k].cuda()

            for k, v in mask_back.items():
                mask_back[k] = mask_back[k].cuda()

    elif 'derpp' in self.args.baseline:
        buffer = memory.Buffer(int(self.args.replay_sample_per_task * self.args.ntasks),args=self.args)
        if self.args.pt_task > 0:
            buffer.load(os.path.join(self.args.output_dir + '../', 'buffer'))

    elif self.args.pt_task > 0 and 'adapter_demix' in self.args.baseline:  # Initialize the new adapter using the nearest adapter
        model = demix.compute(train_loader_subset, train_loader_subset_dataset, model, accelerator,self.args)

    if 'dga' in self.args.baseline or 'das' in self.args.baseline:
        train_loader_prune = accelerator.prepare(train_loader_subset)
        config = accelerator.unwrap_model(model).model.config

        if 'before_distill' in self.args.softmask_compute and (self.args.pt_task == 0 or 'dga' in self.args.baseline): # One and dga are the same

            config = accelerator.unwrap_model(model).model.config
            softmask.compute_impt(args=self.args, config=config, model=model,
                                                 eval_dataloader=train_loader_prune, accelerator=accelerator,
                                                 prune_loss='before_distill')

        if 'before_mlm' in self.args.softmask_compute and self.args.pt_task == 0:  # only for wiki in task 0

            model = accelerator.prepare(model)
            softmask.compute_impt(args=self.args, config=config, model=model,
                                                 eval_dataloader=train_loader_prune, accelerator=accelerator,
                                                 prune_loss='before_mlm')

        accelerator.wait_for_everyone()
        head_impt_accumlate, intermediate_impt_accumlate, output_impt_accumlate = softmask.accumulate_impt(self.args)

        if accelerator.is_main_process:
            print(f'Accumulated head layer importance: {head_impt_accumlate}')
            print(f'Accumulated intermediate layer importance: {intermediate_impt_accumlate}')
            print(f'Accumulated output layer importance: {output_impt_accumlate}')

        if 'head_mask' in self.args.layer_to_mask:
            head_impt = head_impt_accumlate
        if 'intermediate_mask' in self.args.layer_to_mask:
            intermediate_impt = intermediate_impt_accumlate
        if 'output_mask' in self.args.layer_to_mask:
            output_impt = output_impt_accumlate
    
    if 'lora_init' == self.args.baseline and self.args.pt_task > 0:
        train_loader_subset = accelerator.prepare(train_loader_subset)
        for step, inputs in enumerate(tqdm(train_loader_subset)):
            outputs = model(inputs, task_label=self.args.pt_task-1)
            loss = outputs.loss
            
            accelerator.backward(loss)
        
        for module in model.modules():
            if 'LoRAPiggybackLinear' in str(type(module)):
                grad_A = module.lora_As[str(self.args.pt_task-1)].grad.data
                grad_B = module.lora_Bs[str(self.args.pt_task-1)].grad.data
                abs_grad_A = torch.abs(grad_A)
                abs_grad_B = torch.abs(grad_B)
                A_max = torch.max(abs_grad_A)
                B_max = torch.max(abs_grad_B)
                A_impt = abs_grad_A / A_max
                B_impt = abs_grad_B / B_max
                
                module.lora_As[str(self.args.pt_task)].data.copy_(A_impt * module.lora_As[str(self.args.pt_task-1)].data + (1-A_impt) * module.lora_As[str(self.args.pt_task)].data)
                module.lora_Bs[str(self.args.pt_task)].data.copy_(B_impt * module.lora_Bs[str(self.args.pt_task-1)].data + (1-B_impt) * module.lora_Bs[str(self.args.pt_task)].data)
                
                module.lora_As[str(self.args.pt_task-1)].grad = None
                module.lora_Bs[str(self.args.pt_task-1)].grad = None
        accelerator.unwrap_model(model).model.lm_head.dense.classifiers[str(self.args.pt_task-1)].weight.grad = None
        accelerator.unwrap_model(model).model.lm_head.dense.classifiers[str(self.args.pt_task-1)].bias.grad = None
        accelerator.unwrap_model(model).model.lm_head.decoder.classifiers[str(self.args.pt_task-1)].weight.grad = None
        accelerator.unwrap_model(model).model.lm_head.decoder.classifiers[str(self.args.pt_task-1)].bias.grad = None
            

    return self,model,head_impt, intermediate_impt, output_impt,self_fisher,mask_pre,mask_back,buffer


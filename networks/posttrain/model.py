import pdb
import torch
import torch.nn as nn
import utils
from networks import prompt
from networks.baselines import simcse
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput, ModelOutput
from dataclasses import dataclass
import numpy as np


class MyModel(nn.Module):

    def __init__(self, model, teacher=None, args=None):
        super().__init__()
        self.model = model
        self.teacher = teacher
        self.config = model.config
        self.args = args
        self.sim = None
        self.sigmoid = nn.Sigmoid()
        self.mse_loss = nn.MSELoss()
        self.cos = nn.CosineSimilarity()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)
        self.frequency_table = torch.Tensor(
            [1 for _ in range(args.ntasks)]).float().cuda()
        # self.frequency_table = torch.Tensor(
        #     [1 for _ in range(args.ntasks)]).float()
        self.kd_loss = utils.model.DistillKL(1)
        self.dropout = nn.Dropout(0.1)
        self.contrast = utils.model.MyContrastive()

    def forward(self, inputs,
                self_fisher=None,
                masks=None,
                mask_pre=None,
                prune_loss=None, head_mask=None, output_mask=None,  # all for the softmask
                intermediate_mask=None,
                task_label=None
                ):

        input_ids = inputs['input_ids']
        labels = inputs['labels']
        if inputs.get('inputs_ori_ids') is not None:
            inputs_ori_ids = inputs['inputs_ori_ids']
        else:
            inputs_ori_ids = None
        if 't5' in self.args.base_model_name_or_path:
            attention_mask = None
        else:
            attention_mask = inputs['attention_mask']

        contrast_loss = None
        distill_loss = None
        simcse_loss = None
        tacl_loss = None
        taco_loss = None
        infoword_loss = None
        hidden_states = None

        if prune_loss is not None and 'distill' in prune_loss:  # detect importance with KL as L_impt
            outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                 head_mask=head_mask,
                                 output_mask=output_mask,
                                 intermediate_mask=intermediate_mask,
                                 output_hidden_states=True, output_attentions=True)
            teacher_outputs = self.teacher(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                           head_mask=head_mask,
                                           output_mask=output_mask,
                                           intermediate_mask=intermediate_mask,
                                           output_hidden_states=True, output_attentions=True)

            if 'roberta' in self.args.base_model_name_or_path:
                loss = self.kd_loss(
                    teacher_outputs.hidden_states[-1], outputs.hidden_states[-1])
            elif 't5' in self.args.base_model_name_or_path:
                loss = self.kd_loss(
                    teacher_outputs.decoder_hidden_states[-1], outputs.decoder_hidden_states[-1])

        elif prune_loss is not None and 'mlm' in prune_loss:  # detect importance with MLM as L_impt
            outputs = self.model(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,
                                 head_mask=head_mask,
                                 output_mask=output_mask,
                                 intermediate_mask=intermediate_mask,
                                 output_hidden_states=True, output_attentions=True)
            loss = outputs.loss

        else:

            if 'prompt' in self.args.baseline:
                inputs_embeds = prompt.cat_learned_embedding_to_input(
                    self.model, input_ids, self.args.task).cuda()
                labels = prompt.extend_labels(self.model, labels).cuda()
                attention_mask = prompt.extend_attention_mask(
                    self.model, attention_mask).cuda()

                outputs = self.model(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask,
                                     output_hidden_states=True)
            elif ('piggyback' in self.args.baseline or 'lora' in self.args.baseline) and 'Llama' not in self.args.base_model_name_or_path:
                outputs = self.model(input_ids=input_ids, labels=labels,
                                     attention_mask=attention_mask, task_label=task_label, output_hidden_states=True)
            else:
                if 'distill' in self.args.baseline:
                    student_ori = self.model(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,
                                             output_hidden_states=True)

                    teacher_ori = self.teacher(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,
                                               output_hidden_states=True)

                    if 'roberta' in self.args.base_model_name_or_path:
                        distill_loss = self.kd_loss(
                            teacher_ori.hidden_states[-1], student_ori.hidden_states[-1])
                    elif 't5' in self.args.base_model_name_or_path:
                        distill_loss = self.kd_loss(
                            teacher_ori.decoder_hidden_states[-1], student_ori.decoder_hidden_states[-1])
                outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                     output_hidden_states=True)

            loss = outputs.loss

            if 'ewc' in self.args.baseline:
                loss_reg = 0
                if self.args.task > 0:

                    for (name, param), (_, param_old) in zip(self.model.named_parameters(),
                                                             self.teacher.named_parameters()):
                        loss_reg += torch.sum(
                            self_fisher['module.model.' + name] * (param_old.cuda() - param.cuda()).pow(2)) / 2
                loss += self.args.lamb * loss_reg

            elif 'adapter_hat' in self.args.baseline \
                    or 'transformer_hat' in self.args.baseline \
                    or 'adapter_bcl' in self.args.baseline \
                    or 'adapter_classic' in self.args.baseline:  # Transformer hat also need to deal with loss
                reg = 0
                count = 0

                if mask_pre is not None:
                    # for m,mp in zip(masks,self.mask_pre):
                    for key in set(masks.keys()) & set(mask_pre.keys()):
                        m = masks[key]
                        mp = mask_pre[key]
                        aux = 1 - mp
                        reg += (m * aux).sum()
                        count += aux.sum()
                else:
                    for m_key, m_value in masks.items():
                        reg += m_value.sum()
                        count += np.prod(m_value.size()).item()

                reg /= count

                loss += self.args.lamb * reg

                if self.args.task > 0 and 'adapter_classic' in self.args.baseline:
                    pre_pooled_outputs = []
                    cur_task = self.args.task
                    cur_s = self.args.s
                    for pre_t in [x for x in range(cur_task)]:
                        self.args.s = self.args.smax
                        self.args.task = pre_t

                        with torch.no_grad():
                            pre_outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                                     head_mask=head_mask,
                                                     output_mask=output_mask,
                                                     intermediate_mask=intermediate_mask,
                                                     output_hidden_states=True)

                        if 'roberta' in self.args.base_model_name_or_path:
                            pre_pooled_output = pre_outputs.hidden_states[-1]
                        elif 't5' in self.args.base_model_name_or_path:
                            pre_pooled_output = pre_outputs.decoder_hidden_states[-1]
                        mean_pre_pooled_output = torch.mean(
                            pre_pooled_output, dim=1)

                        pre_pooled_outputs.append(
                            mean_pre_pooled_output.unsqueeze(-1).clone())

                    self.args.task = cur_task
                    self.args.s = cur_s

                    pre_pooled_outputs = torch.cat(pre_pooled_outputs, -1)

                    if 'roberta' in self.args.base_model_name_or_path:
                        cur_pooled_outputs = outputs.hidden_states[-1]
                    elif 't5' in self.args.base_model_name_or_path:
                        cur_pooled_outputs = outputs.decoder_hidden_states[-1]
                    mean_cur_pooled_output = torch.mean(
                        cur_pooled_outputs, dim=1)

                    pre_pooled_outputs = torch.cat([pre_pooled_outputs, mean_cur_pooled_output.unsqueeze(-1).clone()],
                                                   -1)  # include itselves

                    pooled_output = self.model.self_attns[self.args.task](
                        pre_pooled_outputs)  # softmax on task
                    pooled_output = pooled_output.sum(-1)  # softmax on task
                    pooled_output = self.dropout(pooled_output)
                    pooled_output = F.normalize(pooled_output, dim=1)

                    mix_pooled_reps = [
                        mean_cur_pooled_output.clone().unsqueeze(1)]
                    mix_pooled_reps.append(pooled_output.unsqueeze(1).clone())
                    cur_mix_outputs = torch.cat(mix_pooled_reps, dim=1)

                    # train attention and contrastive learning at the same time
                    loss += self.contrast(cur_mix_outputs,
                                          con_type='unsupervised')

            elif 'simcse' in self.args.baseline:
                inputs_ori_ids_dup = inputs_ori_ids.repeat(2, 1)
                labels_dup = labels.repeat(2, 1)
                if attention_mask is not None:
                    attention_mask_dup = attention_mask.repeat(2, 1)
                else:
                    attention_mask_dup = None

                outputs_ori = self.model(input_ids=inputs_ori_ids_dup, labels=labels_dup,
                                         attention_mask=attention_mask_dup,
                                         output_hidden_states=True)

                if 'roberta' in self.args.base_model_name_or_path:
                    outputs_ori_hidden_state = outputs_ori.hidden_states[-1].view(
                        -1, 2, 164, 768)
                elif 't5' in self.args.base_model_name_or_path:
                    outputs_ori_hidden_state = outputs_ori.decoder_hidden_states[-1].view(
                        -1, 2, self.args.target_length, 768)

                z1 = outputs_ori_hidden_state[:, 0]
                z2 = outputs_ori_hidden_state[:, 1]
                mean_z1 = torch.mean(z1, dim=1)
                mean_z2 = torch.mean(z2, dim=1)
                simcse_loss = simcse.sequence_level_contrast(mean_z1, mean_z2)

            elif ('dga' in self.args.baseline or 'das' in self.args.baseline) and not prune_loss:
                inputs_ori_ids_dup = inputs_ori_ids.repeat(2, 1)
                labels_dup = labels.repeat(2, 1)
                if attention_mask is not None:
                    attention_mask_dup = attention_mask.repeat(2, 1)
                else:
                    attention_mask_dup = None
                outputs_ori = self.model(input_ids=inputs_ori_ids_dup, labels=labels_dup,
                                         attention_mask=attention_mask_dup,
                                         output_hidden_states=True)

                outputs_pre = self.model(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,
                                         head_mask=head_mask,
                                         intermediate_mask=intermediate_mask,
                                         output_mask=output_mask,
                                         output_hidden_states=True)

                if 'roberta' in self.args.base_model_name_or_path:
                    outputs_ori_hidden_state = outputs_ori.hidden_states[-1].view(
                        -1, 2, 164, 768)
                elif 't5' in self.args.base_model_name_or_path:
                    outputs_ori_hidden_state = outputs_ori.decoder_hidden_states[-1].view(
                        -1, 2, self.args.target_length, 768)

                z1 = outputs_ori_hidden_state[:, 0]
                z2 = outputs_ori_hidden_state[:, 1]
                if 'roberta' in self.args.base_model_name_or_path:
                    z3 = outputs_pre.hidden_states[-1]
                elif 't5' in self.args.base_model_name_or_path:
                    z3 = outputs_pre.decoder_hidden_states[-1]

                mean_z1 = torch.mean(z1, dim=1)
                mean_z2 = torch.mean(z2, dim=1)
                mean_z3 = torch.mean(z3, dim=1)
                contrast_loss = simcse.sequence_level_contrast(
                    mean_z1, mean_z2, mean_z3)

            elif 'tacl' in self.args.baseline and not prune_loss:
                outputs_ori = self.model(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,
                                         output_hidden_states=True)
                outputs_teacher = self.teacher(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,
                                               output_hidden_states=True)

                if 'roberta' in self.args.base_model_name_or_path:
                    z1 = outputs_teacher.hidden_states[-1]  # anchor: masks
                    # positive samples: original
                    z2 = outputs_ori.hidden_states[-1]
                elif 't5' in self.args.base_model_name_or_path:
                    z1 = outputs_teacher.decoder_hidden_states[-1]
                    z2 = outputs_ori.decoder_hidden_states[-1]
                tacl_loss = utils.model.tacl_loss(z1, z2, (labels == -100).long(),
                                                  eps=0.0)  # contrasive_labels: bsz x seqlen; masked positions with 0., otherwise 1.

            elif 'taco' in self.args.baseline and not prune_loss:
                outputs_ori = self.model(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,
                                         output_hidden_states=True)

                if 'roberta' in self.args.base_model_name_or_path:
                    inputs_embeds = getattr(
                        self.model, 'roberta').embeddings(inputs_ori_ids)
                    z1 = outputs_ori.hidden_states[-1]  # anchor: masks
                elif 't5' in self.args.base_model_name_or_path:
                    inputs_embeds = self.model.shared(inputs_ori_ids)
                    z1 = outputs_ori.decoder_hidden_states[-1]

                global_z1 = z1 - inputs_embeds

                bsz, ntoken, nfeature = z1.size()
                # ws: window_size
                ws = 5
                offset = torch.randint(-ws, 0, (bsz, ntoken))
                ids = torch.arange(0, ntoken).expand(bsz, ntoken)

                new_ids = ids + offset
                gloabl_z2 = global_z1[torch.arange(global_z1.shape[0]).unsqueeze(
                    -1), new_ids]  # trick to index a 3D tensor using 2D tensor https://stackoverflow.com/questions/55628014/indexing-a-3d-tensor-using-a-2d-tensor

                global_z1 = F.normalize(global_z1, dim=1)
                gloabl_z2 = F.normalize(gloabl_z2, dim=1)

                taco_loss = utils.model.taco_loss(global_z1, gloabl_z2) * 1e-5

            elif 'infoword' in self.args.baseline and not prune_loss:

                outputs_ori = self.model(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,
                                         output_hidden_states=True)
                outputs_mask = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                          output_hidden_states=True)

                if 'roberta' in self.args.base_model_name_or_path:
                    ngram_z1 = outputs_ori.hidden_states[-1]
                elif 't5' in self.args.base_model_name_or_path:
                    ngram_z1 = outputs_ori.decoder_hidden_states[-1]
                # z1 = ngram_z1[new_ids]# trick to index a 3D tensor using 2D tensor https://stackoverflow.com/questions/55628014/indexing-a-3d-tensor-using-a-2d-tensor
                # z1 = z1.view(ngram_z1.size(0),-1,ngram_z1.size(-1)) # cannot do this, becuase each sequence has a different span

                mean_z1 = []
                for z_id, z in enumerate(ngram_z1):
                    z1 = ngram_z1[z_id][
                        (labels[z_id] != -100).unsqueeze(-1).expand_as(ngram_z1[z_id])]  # Only the span for masked
                    # cannot do this, becuase each sequence has a different span
                    z1 = z1.view(1, -1, ngram_z1.size(-1))
                    mean_z1.append(torch.mean(z1, dim=1))

                mean_z1 = torch.stack(mean_z1).squeeze(1)

                if 'roberta' in self.args.base_model_name_or_path:
                    z2 = outputs_mask.hidden_states[-1]
                elif 't5' in self.args.base_model_name_or_path:
                    z2 = outputs_mask.decoder_hidden_states[-1]
                mean_z2 = torch.mean(z2, dim=1)

                infoword_loss = simcse.sequence_level_contrast(
                    mean_z1, mean_z2)

        return MyRobertaOutput(
            loss=loss,
            contrast_loss=contrast_loss,
            distill_loss=distill_loss,
            simcse_loss=simcse_loss,
            tacl_loss=tacl_loss,
            taco_loss=taco_loss,
            infoword_loss=infoword_loss,
            hidden_states=hidden_states,

        )


@dataclass
class MyRobertaOutput(ModelOutput):
    all_attention: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    contrast_loss: torch.FloatTensor = None
    distill_loss: torch.FloatTensor = None
    simcse_loss: torch.FloatTensor = None
    tacl_loss: torch.FloatTensor = None
    taco_loss: torch.FloatTensor = None
    infoword_loss: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None

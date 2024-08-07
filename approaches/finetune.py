
from utils import utils
from sklearn.metrics import f1_score
import logging
import math
import os
import torch
import wandb
from tqdm.auto import tqdm
from networks import prompt
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    AutoConfig,
    RobertaTokenizer,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    SchedulerType,
    set_seed,
)
import numpy as np

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class Appr(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        return

    # TODO: Multiple-GPU supprt

    def train(self, model, accelerator, train_loader, test_loader):

        # Set the optimizer
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                          weight_decay=self.args.weight_decay)

        num_update_steps_per_epoch = math.ceil(
            len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.epoch * num_update_steps_per_epoch
        else:
            self.args.epoch = math.ceil(
                self.args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        # Prepare everything with the accelerator
        model, optimizer, train_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, test_loader)

        logger.info("***** Running training *****")
        logger.info(
            f"Pretrained Model = {self.args.model_name_or_path},  Dataset name = {self.args.dataset_name}, seed = {self.args.seed}")

        if 'lora' in self.args.baseline:
            os.makedirs(
                f'{self.args.output_dir}/../{self.args.finetune_type}', exist_ok=True)
            summary_path = f'{self.args.output_dir}../{self.args.finetune_type}/{self.args.dataset_name}_finetune_summary'
        else:
            summary_path = f'{self.args.output_dir}../{self.args.dataset_name}_finetune_summary'
        print(f'summary_path: {summary_path}')

        for epoch in range(self.args.epoch):
            print("Epoch {} started".format(epoch))
            train_acc, training_loss = self.train_epoch(
                model, optimizer, train_loader, accelerator, lr_scheduler)
            print("train acc = {:.4f}, training loss = {:.4f}".format(
                train_acc, training_loss))

        micro_f1, macro_f1, acc, test_loss = self.eval(
            model, test_loader, accelerator)

        if self.args.dataset_name in ['chemprot_sup', 'rct_sample_sup']:
            macro_f1 = micro_f1  # we report micro instead

        logger.info(
            "{} On {}, last epoch macro_f1 = {:.4f}, acc = {:.4f} (seed={})".format(self.args.model_name_or_path,
                                                                                    self.args.dataset_name, macro_f1,
                                                                                    acc, self.args.seed))

        if not self.args.hyperparameter_tune:
            if accelerator.is_main_process:
                if 'lora' in self.args.baseline:
                    progressive_f1_path = f'{self.args.output_dir}/../{self.args.finetune_type}/progressive_f1_{self.args.seed}'
                    progressive_acc_path = f'{self.args.output_dir}/../{self.args.finetune_type}/progressive_acc_{self.args.seed}'
                else:
                    progressive_f1_path = f'{self.args.output_dir}/../progressive_f1_{self.args.seed}'
                    progressive_acc_path = f'{self.args.output_dir}/../progressive_acc_{self.args.seed}'

                print(f'Path of progressive f1 score: {progressive_f1_path}')
                print(f'Path of progressive accuracy: {progressive_acc_path}')

                if os.path.exists(progressive_f1_path):
                    f1s = np.loadtxt(progressive_f1_path)
                    accs = np.loadtxt(progressive_acc_path)

                else:
                    f1s = np.zeros(
                        (self.args.ntasks, self.args.ntasks), dtype=np.float32)
                    accs = np.zeros(
                        (self.args.ntasks, self.args.ntasks), dtype=np.float32)

                f1s[self.args.pt_task][self.args.ft_task] = macro_f1
                np.savetxt(progressive_f1_path, f1s, '%.4f', delimiter='\t')

                accs[self.args.pt_task][self.args.ft_task] = acc
                np.savetxt(progressive_acc_path, accs, '%.4f', delimiter='\t')

                if self.args.ft_task == self.args.ntasks - 1:  # last ft task, we need a final one
                    if 'lora' in self.args.baseline:
                        final_f1 = f'{self.args.output_dir}/../{self.args.finetune_type}/f1_{self.args.seed}'
                        final_acc = f'{self.args.output_dir}/../{self.args.finetune_type}/acc_{self.args.seed}'

                        forward_f1 = f'{self.args.output_dir}/../{self.args.finetune_type}/forward_f1_{self.args.seed}'
                        forward_acc = f'{self.args.output_dir}/../{self.args.finetune_type}/forward_acc_{self.args.seed}'
                    else:
                        final_f1 = f'{self.args.output_dir}/../f1_{self.args.seed}'
                        final_acc = f'{self.args.output_dir}/../acc_{self.args.seed}'

                        forward_f1 = f'{self.args.output_dir}/../forward_f1_{self.args.seed}'
                        forward_acc = f'{self.args.output_dir}/../forward_acc_{self.args.seed}'

                    print(f'Final f1 score: {final_f1}')
                    print(f'Final accuracy: {final_acc}')

                    if self.args.baseline == 'one':
                        with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[j][j]) + '\n')
                                f1_file.writelines(str(f1s[j][j]) + '\n')

                    else:
                        with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[-1][j]) + '\n')
                                f1_file.writelines(str(f1s[-1][j]) + '\n')

                        with open(forward_acc, 'w') as file, open(forward_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[j][j]) + '\n')
                                f1_file.writelines(str(f1s[j][j]) + '\n')

    def train_epoch(self, model, optimizer, dataloader, accelerator, lr_scheduler):
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)),
                            disable=not accelerator.is_local_main_process)
        model.train()
        train_acc = 0.0
        training_loss = 0.0
        total_num = 0.0
        for batch, inputs in enumerate(dataloader):
            if 'transformer_hat' in self.args.baseline:
                model_ori = accelerator.unwrap_model(model)
                head_importance, intermediate_importance, output_importance = model_ori.transformer_mask()
                res = model.model(**inputs, head_mask=head_importance, intermediate_mask=intermediate_importance,
                                  output_mask=output_importance)
            elif 'piggyback' in self.args.baseline or 'lora' in self.args.baseline:
                res = model.model(
                    **inputs, task_label=self.args.ft_task, return_dict=True)
            else:
                res = model.model(**inputs, return_dict=True)

            outp = res.logits
            loss = res.loss
            optimizer.zero_grad()
            accelerator.backward(loss)

            if 'lora_piggyback' == self.args.finetune_type:
                for module in model.model.modules():
                    if 'Piggyback' in str(type(module)):
                        # abs_weights_A = module.lora_As[str(self.args.ft_task)].data.abs()
                        # abs_weights_B = module.lora_Bs[str(self.args.ft_task)].data.abs()
                        # module.masks_A[str(self.args.ft_task)].grad.data.div_(
                        #     abs_weights_A.mean())
                        # module.masks_B[str(self.args.ft_task)].grad.data.div_(
                        #     abs_weights_B.mean())
                        abs_lora = module.lora_weight.data.abs()
                        module.masks[str(self.args.ft_task)].grad.data.div_(
                            abs_lora.mean())

            if batch == 0:
                for n, p in accelerator.unwrap_model(model).named_parameters():
                    if p.grad is not None:
                        print('n,p： ', n)

            optimizer.step()
            lr_scheduler.step()

            references = accelerator.gather(inputs['labels'])

            if "roberta" in self.args.base_model_name_or_path or "Llama" in self.args.base_model_name_or_path:
                pred = outp.max(1)[1]
                predictions = accelerator.gather(pred)
                train_acc += (references == predictions).sum().item()
            elif "t5" in self.args.base_model_name_or_path:
                pred = outp.max(2)[1]
                preds = self.args.tokenizer.batch_decode(
                    pred, skip_special_tokens=True)
                labels = self.args.tokenizer.batch_decode(
                    references, skip_special_tokens=True)
                # print(preds)
                # print(labels)

                for pred, label in zip(preds, labels):
                    train_acc += int(pred == label)

            training_loss += loss.item()
            total_num += references.size(0)

            progress_bar.update(1)

        wandb.log({"Train_Loss/Task%s" % (self.args.ft_task): training_loss / total_num,
                   "Train_Acc/Task%s" % (self.args.ft_task): train_acc / total_num})
        # break
        return train_acc / total_num, training_loss / total_num

    def eval(self, model, dataloader, accelerator):
        if self.args.dataset_name == 'restaurant_sup':
            label2idx = {'positive': 0, 'negative': 1, 'neutral': 2}
        elif self.args.dataset_name == 'chemprot_sup':
            label2idx = {'DOWNREGULATOR': 0, 'SUBSTRATE': 1, 'INDIRECT-UPREGULATOR': 2, 'INDIRECT-DOWNREGULATOR': 3,
                         'AGONIST': 4, 'ACTIVATOR': 5, 'PRODUCT-OF': 6, 'AGONIST-ACTIVATOR': 7, 'INHIBITOR': 8,
                         'UPREGULATOR': 9, 'SUBSTRATE_PRODUCT-OF': 10, 'AGONIST-INHIBITOR': 11, 'ANTAGONIST': 12}
        elif self.args.dataset_name == 'aclarc_sup':
            label2idx = {'Uses': 0, 'Future': 1, 'CompareOrContrast': 2,
                         'Motivation': 3, 'Extends': 4, 'Background': 5}
        elif self.args.dataset_name == 'scierc_sup':
            label2idx = {'FEATURE-OF': 0, 'CONJUNCTION': 1, 'EVALUATE-FOR': 2, 'HYPONYM-OF': 3, 'USED-FOR': 4,
                         'PART-OF': 5, 'COMPARE': 6}
        elif self.args.dataset_name == 'camera_sup' or self.args.dataset_name == 'phone_sup':
            label2idx = {'positive': 0, 'negative': 1}

        model.eval()
        label_list = []
        prediction_list = []
        total_loss = 0
        total_num = 0
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)),
                            disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            for batch, inputs in enumerate(dataloader):
                references = accelerator.gather(inputs['labels'])

                if "roberta" in self.args.base_model_name_or_path or "Llama" in self.args.base_model_name_or_path:
                    input_ids = inputs['input_ids']
                    if 'piggyback' in self.args.baseline or 'lora' in self.args.baseline:
                        res = model.model(
                            **inputs, task_label=self.args.ft_task, return_dict=True)
                    else:
                        res = model.model(**inputs, return_dict=True)

                    real_b = input_ids.size(0)
                    loss = res.loss
                    outp = res.logits

                    total_loss += loss.data.cpu().numpy().item()*real_b
                    total_num += real_b
                    if self.args.problem_type != 'multi_label_classification':
                        pred = outp.max(1)[1]
                    else:
                        pred = outp.sigmoid() > 0.5
                    predictions = accelerator.gather(pred)

                elif "t5" in self.args.base_model_name_or_path:
                    input_ids = inputs['input_ids']
                    real_b = input_ids.size(0)
                    if 'piggyback' in self.args.baseline or 'lora' in self.args.baseline:
                        res = model.model.generate(
                            input_ids, task_label=self.args.ft_task, return_dict_in_generate=True, output_scores=True)
                    else:
                        res = model.model.generate(
                            input_ids, return_dict_in_generate=True, output_scores=True)
                    outp = torch.stack(list(res.scores), dim=1)
                    pred = outp.max(2)[1]
                    preds = self.args.tokenizer.batch_decode(
                        pred, skip_special_tokens=True)
                    labels = self.args.tokenizer.batch_decode(
                        references, skip_special_tokens=True)

                    loss_fct = torch.nn.CrossEntropyLoss()
                    max_len = max(outp.shape[1], references.shape[1])
                    outputs = torch.nn.functional.pad(
                        outp, (0, 0, 0, max_len - outp.shape[1]), 'constant', self.args.tokenizer.pad_token_id)
                    target_ids = torch.nn.functional.pad(
                        references, (0, max_len - references.shape[1]), 'constant', self.args.tokenizer.pad_token_id)
                    loss = loss_fct(
                        outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                    total_loss += loss.data.cpu().numpy().item()*real_b
                    total_num += real_b

                    references = torch.Tensor(
                        [label2idx[label] for label in labels])
                    # prediction that is not in the label2idx will be assigned to the incorrect label
                    predictions = torch.Tensor([label2idx.get(pred, len(
                        label2idx) - 1 - label2idx[labels[i]]) for i, pred in enumerate(preds)])

                label_list += references.cpu().numpy().tolist()  # we may use multi-node
                prediction_list += predictions.cpu().numpy().tolist()
                progress_bar.update(1)
                # break

        micro_f1 = f1_score(label_list, prediction_list, average='micro')
        macro_f1 = f1_score(label_list, prediction_list, average='macro')
        accuracy = sum([float(label_list[i] == prediction_list[i])
                       for i in range(len(label_list))]) * 1.0 / len(prediction_list)

        wandb.log({"Eval_Loss/Task%s" % (self.args.ft_task): total_loss/total_num,
                   "Eval_Acc/Task%s" % (self.args.ft_task): accuracy,
                   "Eval_Micro_F1/Task%s" % (self.args.ft_task): micro_f1,
                   "Eval_Macro_F1/Task%s" % (self.args.ft_task): macro_f1, })

        return micro_f1, macro_f1, accuracy, total_loss/total_num

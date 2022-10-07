import json
import os
import torch
import numpy
import datetime
from Tools import ProgressBar, InputIDs_Treatment_Batch
from Loader import Loader_WikiMulti_Raw, ArrangeData
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

cuda_flag = True
if __name__ == '__main__':
    input_data = Loader_WikiMulti_Raw()
    cross_lingual_data = ArrangeData(input_data, ['en', 'fr', 'es', 'it', 'ru', 'de'])
    tokenizer = MT5Tokenizer.from_pretrained('C:/PythonProject/mt5_small/')
    model = MT5ForConditionalGeneration.from_pretrained(
        'C:/PythonProject/WikiMultiExperiment/mT5_Small_Batch/checkpoint-step-500000/')

    save_path = 'C:/PythonProject/mT5_Small_Batch/step-500000-Result/'
    if not os.path.exists(save_path): os.makedirs(save_path)
    ############################################

    if cuda_flag: model = model.cuda()

    batch_size = 1
    model.eval()
    pbar = ProgressBar(n_total=len(cross_lingual_data))
    with torch.set_grad_enabled(False):
        for batch_start in range(0, len(cross_lingual_data), batch_size):
            if os.path.exists(os.path.join(save_path + '%08d.json' % batch_start)):
                continue
            file = open(os.path.join(save_path + '%08d.json' % batch_start), 'w', encoding='UTF-8')
            batch = cross_lingual_data[batch_start:batch_start + batch_size]
            input_ids, attention_mask, summary_ids = InputIDs_Treatment_Batch(batch, tokenizer)
            # loss = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=summary_ids).loss

            result = model.generate(input_ids, num_beams=10, min_length=int(1.0 * summary_ids.size()[1]),
                                    max_length=int(1.2 * summary_ids.size()[1]), repetition_penalty=5.0,
                                    num_return_sequences=10)

            pbar(batch_start)

            assembly_result = {}
            for key in batch[0]:
                assembly_result[key] = batch[0][key]
            assembly_result['predict'] = []
            for sample in tokenizer.batch_decode(result, skip_special_tokens=True):
                assembly_result['predict'].append(sample)

            json.dump(assembly_result, file)
            # exit()
            # tensor(2.4974, device='cuda:0')

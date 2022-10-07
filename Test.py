import json
import os
import torch
import numpy
import datetime
from Tools import ProgressBar, InputIDs_Treatment_Batch, InputIDs_Treatment_MultiRoad
from Loader import Loader_WikiMulti_Raw, ArrangeData, ArrangeData_MultiLingualMatch_AllSeparate
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from Model.MT5ForMultiRoad_StackAttentionType import MT5ForMultiRoad_StackAttentionType

cuda_flag = True
if __name__ == '__main__':
    input_data = Loader_WikiMulti_Raw()
    # cross_lingual_data = ArrangeData(input_data, ['en', 'fr', 'es', 'it', 'ru', 'de'])
    cross_lingual_data = ArrangeData_MultiLingualMatch_AllSeparate(input_data, ['en', 'fr', 'es', 'it', 'ru', 'de'])

    # Etymology and scope The word agriculture is a late Middle English

    tokenizer = MT5Tokenizer.from_pretrained('C:/PythonProject/mt5_small/')
    model = MT5ForMultiRoad_StackAttentionType.from_pretrained(
        'C:/PythonProject/mT5_Small_Batch/checkpoint-step-500000/')

    if cuda_flag: model = model.cuda()

    batch_size = 1
    model.eval()
    with torch.set_grad_enabled(False):
        for batch_start in range(0, len(cross_lingual_data), batch_size):
            batch = cross_lingual_data[batch_start:batch_start + batch_size]
            input_ids, attention_mask, summary_ids = InputIDs_Treatment_MultiRoad(batch, tokenizer)
            loss = model.forward(input_ids=input_ids, labels=summary_ids).loss
            print(loss)
            exit()
            # tensor(2.4974, device='cuda:0')

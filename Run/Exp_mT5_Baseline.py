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
    numpy.random.shuffle(cross_lingual_data)

    train_data, test_data = [], []
    for index in range(len(cross_lingual_data)):
        if index % 20 == 0:
            test_data.append(cross_lingual_data[index])
        else:
            train_data.append(cross_lingual_data[index])

    tokenizer = MT5Tokenizer.from_pretrained('C:/PythonProject/mt5_small/')
    save_path = 'mT5_Small_Batch/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    model = MT5ForConditionalGeneration.from_pretrained('C:/PythonProject/mt5_small/')
    ############################################

    if cuda_flag: model = model.cuda()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1E-4)

    batch_size = 4
    total_loss = 0.0
    step_counter = 0
    model.zero_grad()
    pbar = ProgressBar(n_total=20 * len(train_data))
    for epoch in range(20):
        for batch_start in range(0, len(train_data), batch_size):
            batch = train_data[batch_start:batch_start + batch_size]
            step_counter += 1
            input_ids, attention_mask, summary_ids = InputIDs_Treatment_Batch(batch, tokenizer)
            loss = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=summary_ids).loss
            loss.backward()
            total_loss += loss.data

            optimizer.step()
            model.zero_grad()
            pbar(step_counter, {'loss': loss.data})
            if step_counter % 1 == 0:
                print("\nstep: %7d\t loss: %7f\n" % (step_counter, total_loss))
                total_loss = 0.0

                with torch.set_grad_enabled(False):
                    model.eval()
                    val_pbar = ProgressBar(n_total=len(test_data))
                    for batch_start in range(0, len(test_data), batch_size):
                        batch = test_data[batch_start:batch_start + batch_size]
                        input_ids, attention_mask, summary_ids = InputIDs_Treatment_Batch(batch, tokenizer)
                        loss = model.forward(input_ids=input_ids, attention_mask=attention_mask,
                                             labels=summary_ids).loss

                        val_pbar(batch_start, {'loss': loss.data})
                        total_loss += loss.item()
                    print('\nVal Part Loss = ', total_loss)
                    with open(os.path.join(save_path, "log"), "a", encoding="UTF-8") as log:
                        log.write(
                            "%s\t step: %6d\t loss: %.2f\t \n" % (datetime.datetime.now(), step_counter, total_loss))

                    filename = "checkpoint-step-%06d" % step_counter
                    full_filename = os.path.join(save_path, filename)
                    model.save_pretrained(save_path + filename)

                model.train()
                total_loss = 0.0

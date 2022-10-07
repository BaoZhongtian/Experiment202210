import time
import torch
import numpy


def InputIDs_Treatment(batch, tokenizer, cuda_flag=True, max_length=2048):
    if batch is None: return torch.LongTensor([[1]]).cuda(), None

    article = batch['CrossLingualDocument']
    english_summary = batch['EnglishSummary']

    treat_lingual = batch['CrossLingualName']

    if treat_lingual == 'spanish':
        input_ids = tokenizer.batch_encode_plus(['Summarize Spanish to English :' + article],
                                                max_length=max_length, return_tensors='pt', truncation=True)[
            'input_ids']
    if treat_lingual == 'portuguese':
        input_ids = tokenizer.batch_encode_plus(['Summarize Portuguese to English :' + article],
                                                max_length=max_length, return_tensors='pt', truncation=True)[
            'input_ids']
    if treat_lingual == 'turkish':
        input_ids = tokenizer.batch_encode_plus(['Summarize Turkish to English :' + article],
                                                max_length=max_length, return_tensors='pt', truncation=True)[
            'input_ids']
    if treat_lingual == 'russian':
        input_ids = tokenizer.batch_encode_plus(['Summarize Russian to English :' + article],
                                                max_length=max_length, return_tensors='pt', truncation=True)[
            'input_ids']
    if treat_lingual == 'vietnamese':
        input_ids = tokenizer.batch_encode_plus(['Summarize Vietnamese to English :' + article],
                                                max_length=max_length, return_tensors='pt', truncation=True)[
            'input_ids']
    if treat_lingual == 'english':
        input_ids = tokenizer.batch_encode_plus(['Summarize English to English :' + article],
                                                max_length=max_length, return_tensors='pt', truncation=True)[
            'input_ids']
    assert input_ids is not None

    summary_ids = tokenizer.batch_encode_plus([english_summary], return_tensors='pt')['input_ids']
    if cuda_flag:
        input_ids, summary_ids = input_ids.cuda(), summary_ids.cuda()
    return input_ids, summary_ids


def InputIDs_Treatment_Batch(batch, tokenizer, cuda_flag=True, max_length=2048):
    if batch is None: return torch.LongTensor([[1]]).cuda(), None
    total_document, total_summary = [], []
    for sample in batch:
        if 'en_text' in sample: total_document.append('Summarize English to English : ' + sample['en_text'])
        if 'fr_text' in sample: total_document.append('Summarize French to English : ' + sample['fr_text'])
        if 'es_text' in sample: total_document.append('Summarize Spanish to English : ' + sample['es_text'])
        if 'it_text' in sample: total_document.append('Summarize Italian to English : ' + sample['it_text'])
        if 'ru_text' in sample: total_document.append('Summarize Russian to English : ' + sample['ru_text'])
        if 'de_text' in sample: total_document.append('Summarize German to English : ' + sample['de_text'])
        total_summary.append(sample['en_summary'])

    input_raw = tokenizer.batch_encode_plus(
        total_document, return_tensors='pt', padding=True, max_length=max_length, truncation=True)
    summary_ids = tokenizer.batch_encode_plus(
        total_summary, return_tensors='pt', padding=True, max_length=max_length, truncation=True)['input_ids']
    for index_x in range(summary_ids.size()[0]):
        for index_y in range(summary_ids.size()[1]):
            if summary_ids[index_x][index_y] == 0: summary_ids[index_x][index_y] = -100

    input_ids, attention_mask = input_raw['input_ids'], input_raw['attention_mask']
    if cuda_flag:
        input_ids, attention_mask, summary_ids = input_ids.cuda(), attention_mask.cuda(), summary_ids.cuda()
    return input_ids, attention_mask, summary_ids


def InputIDs_Treatment_MultiRoad(input_dict, tokenizer, cuda_flag=True, max_length=2048):
    if input_dict is None: return torch.LongTensor([[1]]).cuda(), None
    assert len(input_dict) == 1
    input_dict = input_dict[0]
    english_summary = input_dict['en_summary']

    total_document = []
    if 'en_text' in input_dict: total_document.append('Summarize English to English : ' + input_dict['en_text'])
    if 'fr_text' in input_dict: total_document.append('Summarize French to English : ' + input_dict['fr_text'])
    if 'es_text' in input_dict: total_document.append('Summarize Spanish to English : ' + input_dict['es_text'])
    if 'it_text' in input_dict: total_document.append('Summarize Italian to English : ' + input_dict['it_text'])
    if 'ru_text' in input_dict: total_document.append('Summarize Russian to English : ' + input_dict['ru_text'])
    if 'de_text' in input_dict: total_document.append('Summarize German to English : ' + input_dict['de_text'])

    input_ids = tokenizer.batch_encode_plus(
        total_document, return_tensors='pt', max_length=max_length, truncation=True)['input_ids']
    summary_ids = tokenizer.batch_encode_plus([english_summary], return_tensors='pt')['input_ids']
    if cuda_flag:
        input_ids = [_.unsqueeze(0).cuda() for _ in input_ids]
        summary_ids = summary_ids.cuda()
    return input_ids, None, summary_ids


class ProgressBar(object):
    def __init__(self, n_total, width=30, desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')

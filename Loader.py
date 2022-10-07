import os
import tqdm
import pandas


def Loader_WikiMulti_Raw():
    load_path = 'C:/PythonProject/MoreDataset/wikimulti.csv'
    data_raw = pandas.read_csv(load_path)
    # print(data_raw.columns)
    return data_raw


def ArrangeData(input_data, appoint_lingual=[], max_sample_number=10):
    total_sample = []
    for row_index in tqdm.trange(input_data.shape[0]):
        for lingual_name in appoint_lingual:
            if pandas.isnull(input_data.iloc[row_index]['%s_text' % lingual_name]): continue

            total_sample.append(
                {'title_en': input_data.iloc[row_index]['title_en'],
                 'en_summary': input_data.iloc[row_index]['en_summary'].replace('\n', ' '),
                 '%s_text' % lingual_name: input_data.iloc[row_index]['%s_text' % lingual_name].replace('\n', ' ')})

            if len(total_sample) > max_sample_number: break
    return total_sample


def ArrangeData_MultiLingualMatch_AllSeparate(input_data, appoint_lingual=[], max_sample_number=10):
    total_sample = []
    for row_index in tqdm.trange(input_data.shape[0]):
        sample_lingual = {}
        for lingual_name in appoint_lingual:
            if pandas.isnull(input_data.iloc[row_index]['%s_text' % lingual_name]): continue
            sample_lingual['%s_text' % lingual_name] = input_data.iloc[row_index]['%s_text' % lingual_name].replace(
                '\n', ' ')

            # total_sample.append(
            #     {'title_en': input_data.iloc[row_index]['title_en'],
            #      'en_summary': input_data.iloc[row_index]['en_summary'].replace('\n', ' '),
            #      '%s_text' % lingual_name: input_data.iloc[row_index]['%s_text' % lingual_name].replace('\n', ' ')})

        lingual_used = [_ for _ in sample_lingual.keys()]

        if len(lingual_used) >= 2:
            for index_x in range(len(lingual_used)):
                for index_y in range(index_x + 1, len(lingual_used)):
                    total_sample.append(
                        {'title_en': input_data.iloc[row_index]['title_en'],
                         'en_summary': input_data.iloc[row_index]['en_summary'].replace('\n', ' '),
                         lingual_used[index_x]: sample_lingual[lingual_used[index_x]],
                         lingual_used[index_y]: sample_lingual[lingual_used[index_y]]})

        # if len(lingual_used) >= 3:
        #     for index_x in range(len(lingual_used)):
        #         for index_y in range(index_x + 1, len(lingual_used)):
        #             for index_z in range(index_y + 1, len(lingual_used)):
        #                 total_sample.append(
        #                     {'title_en': input_data.iloc[row_index]['title_en'],
        #                      'en_summary': input_data.iloc[row_index]['en_summary'].replace('\n', ' '),
        #                      lingual_used[index_x]: sample_lingual[lingual_used[index_x]],
        #                      lingual_used[index_y]: sample_lingual[lingual_used[index_y]],
        #                      lingual_used[index_z]: sample_lingual[lingual_used[index_z]]})

        if len(total_sample) > max_sample_number: break

    print(len(total_sample))
    return total_sample
    # One 84532
    # Two 239183
    # Three 408999


if __name__ == '__main__':
    input_data = Loader_WikiMulti_Raw()
    ArrangeData_MultiLingualMatch_AllSeparate(input_data, ['en', 'fr', 'es', 'it', 'ru', 'de'])

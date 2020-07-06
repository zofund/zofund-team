import os
import re
import pandas as pd
from source.LocalEnv import INPUT_PATH


class DataPreprocessor:
    @staticmethod
    def text_cleaning(lines):
        lines = lines.replace('\u3000', '').replace('null', '').strip()  # 去除多余的空格和null
        pattern = re.compile('<(.*?)>')  # 去除文中的<>标签(包括AI标签，以及多余的文末总结)
        res = re.sub(pattern, '', lines)

        return res

    @staticmethod
    def df_cleaning(df):
        """
        输入原始数据的DataFrame，对数据进行初步清洗
        :param df:
        :return:
        """
        df.loc[:, 'report_summary'] = df['report_summary'].apply(DataPreprocessor.text_cleaning)
        return df


if __name__ == '__main__':
    data = pd.read_csv(os.path.join(INPUT_PATH, 'raw_data.csv'))
    data = DataPreprocessor.df_cleaning(data)
    data.to_csv(os.path.join(INPUT_PATH, 'clean_data.csv'), encoding='utf-8-sig', index=False)

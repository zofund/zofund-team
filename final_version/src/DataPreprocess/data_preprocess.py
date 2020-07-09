import os
import re
import pandas as pd
import numpy as np
from src.LocalEnv import INPUT_PATH


class DataPreprocessor:
    @staticmethod
    def text_cleaning(lines):
        lines = lines.replace('\u3000', '').replace('null', '').strip()  # 去除多余的空格和null
        pattern = re.compile('<(.*?)>')  # 去除文中的<>标签(包括AI标签，以及多余的文末总结)
        res = re.sub(pattern, '', lines)

        return res

    @staticmethod
    def _adding_report_name(x):
        """
        由于report的标题往往是非常具有总结性的内容，可能包含了大量有用信息，将report name添加至report summary中
        :param x:
        :return:
        """
        report_name = x['report_name']
        if ':' in report_name or '：' in report_name:
            ret = x['report_summary'] + report_name.split(':')[-1].split('：')[-1]
        else:
            return np.nan
        return ret

    @staticmethod
    def df_cleaning(df):
        """
        输入原始数据的DataFrame，对数据进行初步清洗,
        包含以下几个步骤:
        1. 清除<AI>标签，清除其他多余的<>标签，清除多余的null等。
        2. 添加文章标题。
        3. 清除无意义的report summary
        :param df:
        :return:
        """
        df.loc[:, 'report_summary'] = df['report_summary'].apply(DataPreprocessor.text_cleaning)
        df.loc[:, 'report_summary'] = df.apply(DataPreprocessor._adding_report_name, axis=1)
        df = df.loc[~df['report_summary'].isna(), :]
        return df


if __name__ == '__main__':
    data = pd.read_csv(os.path.join(INPUT_PATH, 'raw_data.csv'))
    print('原始数据有{}条'.format(len(data)))
    data = DataPreprocessor.df_cleaning(data)
    data.to_csv(os.path.join(INPUT_PATH, 'clean_data.csv'), encoding='utf-8-sig', index=False)
    print('处理后的数据有{}条'.format(len(data)))

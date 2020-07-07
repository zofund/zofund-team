import configparser
import psycopg2
import os
import pandas as pd
from src.LocalEnv import SETTING_PATH, INPUT_PATH


class GetRawData:
    def __init__(self):
        cp = configparser.ConfigParser()
        cp.read(SETTING_PATH)
        server = cp['intern']['server']
        port = cp['intern']['port_number']
        instance_ = cp['intern']['instance']
        username = cp['intern']['username']
        password = cp['intern']['password']

        self.conn = psycopg2.connect(database=instance_, user=username, password=password, host=server, port=port)
        self.cursor = self.conn.cursor()

    def get_data_all(self):
        """
        从数据库读取所有数据
        :return:
        """
        sql_data = """
        select * from intern_test
        """
        self.cursor.execute(sql_data)
        data = self.cursor.fetchall()
        df_ = pd.DataFrame.from_records(data)

        sql_col = """
        select column_name from information_schema.columns
        where table_schema='public' and table_name='intern_test'
        """
        self.cursor.execute(sql_col)
        columns = self.cursor.fetchall()
        columns = [i[0] for i in columns]
        df_.columns = columns

        return df_

    def save_(self, data, file_name):
        """
        保存df为csv文件
        :param data:
        :param file_name:
        :return:
        """
        path = os.path.join(INPUT_PATH, file_name + '.csv')
        data.to_csv(path, encoding='utf-8-sig', index=False)


if __name__ == '__main__':
    gwd = GetRawData()
    df = gwd.get_data_all()
    gwd.save_(df, 'raw_data')


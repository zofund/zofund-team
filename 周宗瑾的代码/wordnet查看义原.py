# import tensorflow as tf
#
# from bert_serving.server.helper import get_args_parser
#
# from bert_serving.server import BertServer
#
#
# hello = tf.constant('hello')
# sess = tf.Session()
# print(sess.run(hello))

# args = get_args_parser().parse_args(['-model_dir', 'chinese_L-12_H-768_A-12',
#                                      '-port', '15555', # 客户端连接填入port和port_out参数
#                                      '-port_out', '15556',
#                                      '-max_seq_len', 'NONE',
#                                      '-mask_cls_sep',
#                                      '-cpu'])
#
# if __name__ == '__main__':
#     server = BertServer(args)
#     server.start()
#
#
# # -*- coding: utf-8 -*-
# import numpy as np
# from bert_serving.client import BertClient
#
# print('开始运行')
# bc = BertClient(port=15555, port_out=15556)
# print('开始连接')
# result = bc.encode(['人工 智能','数据 挖掘'])
# print(result)
# print(result.shape)


from nlkt.corpus import wordnet as wn
a = wn.synsets('升高')
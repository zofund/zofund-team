#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 15:11
# @Author  : Boting CHEN
# @Site    : 
# @File    : LocalEnv.py
# @Software: PyCharm

import os
from os.path import join
#  --------------------------路径相关------------------------------
# 项目目录
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# source目录
SRC_PATH = join(PROJECT_PATH, 'src/')
# 配置文件目录
SETTING_PATH = join(PROJECT_PATH, 'Setting.conf')
# Input Path
INPUT_PATH = join(PROJECT_PATH, 'data', 'input')
# Output Path
OUTPUT_PATH = join(PROJECT_PATH, 'data', 'output')
# Log Dir
LOG_DIR = join(PROJECT_PATH, 'data', 'logging')
# Saved Model Path
SAVED_MODEL_PATH = join(PROJECT_PATH, 'data', 'saved_model')


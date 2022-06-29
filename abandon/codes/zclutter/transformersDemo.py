# coding=utf-8
# Copyright (C) xxx team - All Rights Reserved
# 
# @Version:   3.9.5
# @Software:  PyCharm
# @FileName:  transformersDemo.py
# @CTime:     2022/6/29 23:10   
# @Author:    yhy
# @Email:     yhy@yhy.com
# @UTime:     2022/6/29 23:10
# 
# @Description:
# 
#     xxx
# 
import codecs
import logging
from typing import List, Dict, Optional
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


def main():
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base", mirror='tuna')
    model = AutoModel.from_pretrained("hfl/chinese-macbert-base", mirror='tuna')

    inputs = tokenizer("Hello world!你好，我是中国人！", return_tensors="pt")
    outputs = model(**inputs)

    last_hidden_state, pooler_output = outputs.to_tuple()

    print(last_hidden_state.shape, pooler_output.shape)



if __name__ == '__main__':
    main()

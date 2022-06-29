# coding=utf-8
# Copyright (C) xxx team - All Rights Reserved
# 
# @Version:   3.9.5
# @Software:  PyCharm
# @FileName:  cos_similarity_test.py.py
# @CTime:     2022/6/29 23:13   
# @Author:    yhy
# @Email:     yhy@yhy.com
# @UTime:     2022/6/29 23:13
# 
# @Description:
# 
#     xxx
# 
import codecs
import logging
from typing import List, Dict, Optional
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def main(model_name, text):
    bert = AutoModel.from_pretrained(model_name)
    bert.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = tokenizer.batch_encode_plus(text, padding=True, return_tensors='pt')
    outputs = bert(text['input_ids'], attention_mask=text['attention_mask'])

    return {
        'cls': np.around(cosine_similarity(outputs['last_hidden_state'][:, 0, :].detach().numpy()), 4),
        'mean': np.around(cosine_similarity(torch.mean(outputs['last_hidden_state'], dim=-2).detach().numpy()), 4),
    }



if __name__ == '__main__':
    model_name = 'hfl/rbt3'
    text = [
        '我爱你！',
        '我也爱你',
        '不我爱你',
        '我不爱你',
        '我爱不你',
        '我爱你不',
    ]

    output = main(model_name, text)
    print(f"cls cosine_similarity:\n {output['cls']}\n\n"
          f"mean cosine_similarity:\n {output['mean']}")

    # cls cosine_similarity:
    #  [[1.     0.8985 0.8868 0.8824 0.878  0.7617]
    #  [0.8985 1.     0.959  0.9709 0.9479 0.8038]
    #  [0.8868 0.959  1.     0.9808 0.9688 0.8274]
    #  [0.8824 0.9709 0.9808 1.     0.9708 0.8263]
    #  [0.878  0.9479 0.9688 0.9708 1.     0.8742]
    #  [0.7617 0.8038 0.8274 0.8263 0.8742 1.    ]]
    #
    # mean cosine_similarity:
    #  [[1.     0.8836 0.8717 0.8608 0.8611 0.833 ]
    #  [0.8836 1.     0.9375 0.9491 0.9169 0.8766]
    #  [0.8717 0.9375 1.     0.9742 0.9636 0.8981]
    #  [0.8608 0.9491 0.9742 1.     0.9612 0.8999]
    #  [0.8611 0.9169 0.9636 0.9612 1.     0.9208]
    #  [0.833  0.8766 0.8981 0.8999 0.9208 1.    ]]

    # 有个结论，bert预训练出来的向量都很相似

import logging
import time
import json
import requests
from requests.exceptions import RequestException
from tqdm import tqdm

import torch
import torch.nn.functional as F

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from typing import List, Optional, Tuple, Type, TypeVar
logger = logging.getLogger(__name__)
import MNN.expr as expr
import MNN.numpy as np
import MNN.nn as nn


@register_model("mnn")
class MNN(LM):
    def __init__(self, mnn_model=None, tokenizer = None, **kwargs):
        super().__init__()
        self.mnn_model = mnn_model
        self.tokenizer = tokenizer
        self.logprobs = 10
        # self.temperature = 0.0
        # self.max_length = max_length


    def _encode_pair(self, context, continuation):
        """
        对给定的上下文和后续内容进行编码。
        :param context: 上下文文本字符串。
        :param continuation: 继续的文本字符串。
        :return: 上下文编码和后续编码的元组。
        """
        # 计算context字符串末尾的空格数，并在continuation中添加相应数目的空格，同时更新context。空格移动到continuation里面了
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        # 对整个上下文和后续内容进行编码，然后分割获取上下文和后续的单独编码
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc
    def tok_encode(self,string):
        encoding = self.tokenizer.encode(string, add_special_tokens=False)
        encoding = np.array(encoding)
        return encoding #这是一个var列表
    
    def loglikelihood(self, requests, disable_tqdm: bool = False)-> List[Tuple[float, bool]]:
        """
        输入：Tuple[str,str],
            str1:input string to the LM;
            str2:a target string
        输出：Tuple[float,int]
            ll:似然比率
            is_greedy:0/1
        """
        # 如果请求列表为空，则直接返回空列表
        res = []
        # 使用tqdm为每个请求的参数对进行迭代，计算其对数似然
        for context, continuation in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):
            # 调用模型完成函数，这里直接调用本地模型得到答案就行。
            context_enc, continuation_enc = self._encode_pair(context, continuation)
            context_len = len(context_enc)
            continuation_len = len(continuation_enc)
            #获取输入,拼接，截断
            max_length = 127
            # inp = np.concatenate((context_enc, continuation_enc), axis=0)[-(max_length +1):][:-1]
            inp = np.concatenate((context_enc, continuation_enc), axis=0)[:-1]
            inplen = len(inp)
            #计算全部似然
            logprobs = self.mnn_model.forward(inp)
            #转化成torch,并且softmax
            logprobs = torch.from_numpy(logprobs.read())
            logits = F.log_softmax(logprobs,dim = -1)
            #获取continuation的logits并求和
            #这里考虑截取
            continuation_enc = torch.from_numpy(continuation_enc.read()).to(torch.long).unsqueeze(0)#[1,seq]
            logits = logits[:,inplen - continuation_len : inplen,:]
            logits = torch.gather(logits, 2, continuation_enc.unsqueeze(-1)).squeeze( -1)
            logprob = float(logits.sum())
            is_greedy = False
            res.append((logprob, is_greedy))
            with open('mnn_qa.txt','a') as f:
                f.write(f"问题是:{context} >>> 回答是:{continuation} >>> logprob是:{logprob}\n")
        # print(res)
        with open('mnn_res.json', 'w') as f:
            json.dump(res, f)
        return res
    
    #piqa应该是属于generate_until
    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "generate not yet supported for GGUF models"
        )
    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )

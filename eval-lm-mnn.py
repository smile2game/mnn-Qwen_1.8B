from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval.models.mnn import MNN
from lm_eval import simple_evaluate, tasks
import json



#######################用于加载模型#######################


import os
import glob
import argparse
import onnxruntime as ort
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import MNN.nn as nn
import MNN.expr as F
import MNN.numpy as np
import numpy
import torch
import time
from transformers import AutoTokenizer

#没有指定父类，所以不需要写()
#这个类是基础
class LLM:
    def __init__(self, model_path):
        self.max_length = 2048
        self.load(model_path)
        self.context_len = 0
        self.token_len = 0
        self.past_kv_shape = [2, 1, 0, 16, 128]
    def load_module(self, path, name, inputs=[], outputs=[]):
        return nn.load_module_from_file(os.path.join(path, name), inputs, outputs,
                                        precision_mode = F.PrecisionMode.Low,
                                        memory_mode = F.MemoryMode.Low,
                                        backend = F.Backend.CPU,
                                        rearrange = True,
                                        shape_mutable = True
                                        )

    def load(self, model_path):
        # load split
        self.block_nums = len(glob.glob(os.path.join(model_path, 'block_*.mnn')))
        self.lm = self.load_module(model_path,'lm.mnn',)
        print("load lm没问题啦,这里返回值是m_logits!!\n")
        self.embed = self.load_module(model_path, 'embedding.mnn')
        self.blocks = [None for i in range(self.block_nums)]
        for i in range(self.block_nums):
            self.blocks[i] = self.load_module(model_path, f'block_{i}.mnn',
                                              ["inputs_embeds", "attention_mask", "position_ids", "past_key_values"],
                                              ["hidden_states", "presents"])
    def get_attention_mask(self) -> F.Var:
        if self.token_len:
            return np.ones([1, 1, 1, 1])
        return np.array(numpy.tril(numpy.ones([1, 1, self.seq_len, self.seq_len], dtype=numpy.int32)).tolist())

    def get_position_ids(self) -> F.Var:
        if self.token_len:
            return np.array([self.seq_len - 1])
        return np.arange(self.seq_len, dtype=np.int32)
    
    def stop_id(self):
        return self.tokenizer.im_end_id


    def forward(self, input_ids):
        self.seq_len = input_ids.size
        self.context_len = self.seq_len - 2
        #这里可能也会造成问题
        self.token_len = 0
        attention_mask = self.get_attention_mask()
        position_ids = self.get_position_ids()
        past_key_values = [F.placeholder(self.past_kv_shape, dtype=np.float32) for i in range(self.block_nums)]
        hidden_states = self.embed(input_ids)
        
        presents = []

        for i in range(self.block_nums):
            hidden_states, kv = self.blocks[i]([hidden_states, attention_mask, position_ids, past_key_values[i]])
            presents.append(kv)

        #mnn推理lm部分，这部分没问题了。
        m_logits = self.lm(hidden_states) 
        return m_logits

model_path = "./data/final-qwen-logits"
tokenizer_path = "./data/Qwen-1_8B-Chat"

# 加载预训练模型和tokenizer
# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval()
model  = LLM(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# 使用HFLM包装模型和tokenizer，设置批处理大小和设备
# lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=64, device="cpu")

#这是核心
lm = MNN(mnn_model = model,tokenizer = tokenizer)
# 初始化任务
# tasks.initialize_tasks()

# 简单评估模型在指定任务上的性能
results = simple_evaluate(model=lm, tasks=["piqa"])

# 将评估结果中的数据导出到JSON文件中
filtered_results = {key: value for key, value in results.items() if key != "samples"}  # 过滤掉不需要的数据
json_filtered_results = json.dumps(filtered_results, indent=4)  # 转换为JSON格式并添加缩进

with open("results_mnn.json", "w") as json_file:
    json_file.write(json_filtered_results)  # 写入到JSON文件中

"""
-*- coding: utf-8 -*-
2018/11/6 11:26 rouge155.py
@Author: HL
@E-mail: minorfox@qq.com

 In ROUGE parlance, your summaries are 'system' summaries and the gold standard summaries are 'model' summaries.
 The summaries should be in separate folders, whose paths are set with the system_dir and model_dir variables.
 All summaries should contain one sentence per line.
"""

from pyrouge import Rouge155
import os

def compute_rouge(sentences, targets):
    target_path = "./dir/target"
    senten_path = "./dir/senten"
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    if not os.path.exists(senten_path):
        os.mkdir(senten_path)
    count = 1
    for sent, tgt in zip(sentences, targets):
        with open(target_path+"/text.A."+str(count)+".txt", mode='w', encoding='utf-8') as f:
            with open(senten_path+"/text."+str(count)+".txt", mode='w', encoding='utf-8') as g:
                count += 1
                # print(sent)
                # print(tgt)
                for s in sent:
                    g.write(str(s)+" ")
                for t in tgt:
                    f.write(str(t)+" ")

    r = Rouge155()
    r.system_dir = senten_path
    r.model_dir = target_path
    r.system_filename_pattern = 'text.(\d+).txt'
    r.model_filename_pattern = 'text.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate()
    # print(output)
    output_dict = r.output_to_dict(output)
    rg1 = output_dict["rouge_1_f_score"]
    rg2 = output_dict["rouge_2_f_score"]
    rgl = output_dict["rouge_l_f_score"]

    return rg1, rg2, rgl




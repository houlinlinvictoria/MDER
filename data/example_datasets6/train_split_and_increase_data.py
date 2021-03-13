import xlrd
from itertools import chain
import random
from random import choice
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time

time1 = time.time()

"""创建小写字母词表"""
text_sens_data = xlrd.open_workbook('AAAI', 'rb').sheet_by_name(u'Sheet1')
text_data = xlrd.open_workbook('Y_Datasets.xlsx', 'rb').sheet_by_name(u'Sheet1')
text_method = xlrd.open_workbook('Y_Methods.xlsx', 'rb').sheet_by_name(u'Sheet1')
text_none = xlrd.open_workbook('N_MandD.xlsx', 'rb').sheet_by_name(u'Sheet1')
data_dic = text_data.col_values(0) #数据集白名单
method_dic = text_method.col_values(0) #方法白名单
none_dic = text_none.col_values(0) #普通词黑名单
sens_row = text_sens_data.nrows #2800，句子总数
print(sens_row)

sens = text_sens_data.col_values(2) #带标记的句子表，第三列
#add_m = '~'
#add_d = '^'
#add_n = '`' #特殊字符中间夹着的为黑名单，论文里的句子不能出现这三个符号
#dot = ','
#
#for i in range(len(data_dic)):
#    data_dic[i] = data_dic[i].replace(' ', ',')
#for i in range(len(method_dic)):
#    method_dic[i] = method_dic[i].replace(' ', ',')
#for i in range(len(none_dic)):
#    none_dic[i] = none_dic[i].replace(' ', ',')
#
#
#for row in range(sens_row): #句子清理后带标记的全部句子为sens list
#    sens[row] = sens[row].replace('\t', '').replace('- ', '-').replace(' - ', '-').replace(' -', '-').replace('+ ', '+').replace(';',' ;').replace(
#        ' + ', '+').replace(' +', '+').replace('  ', ' ').replace('.', ' .').replace(',', ' ,').replace(':', ' :').replace('(', '( ').replace(')', ' )').replace(' ', ',')  # re.sub('”', '\"', sens[row])

"""前2800组数据带方法集和数据集实体的句子拿出来"""    
def entity_sentence(sentence): #sentence为带标记的表第三列，经过句子清理的
    MD_sentence = [] #含M或者D数量
    for i in range(len(sentence)):
        m = '~'
        d = '^'
        a = m in sentence[i]
        b = d in sentence[i]
        if a or b: # 如果a 或者 b 有一个是true
            MD_sentence.append(sentence[i])
    print('含有实体的句子总数：',len(MD_sentence)) 
    return MD_sentence
MD_sentences = entity_sentence(sens)


"""数据中方法集和数据集实体数目和具体词典"""
def number_M_D(sens):
    method_list = []
    dataset_list = []
    for row in range(len(sens)):
        sentence = sens[row]
        num = 0
        while num < len(sentence):
            if sentence[num] == '~':
                num += 1
                start = num
                count = 0
                while (sentence[num] != '~'):
                    num += 1
                    count += 1
                end = start + count
                num += 1
                method_data = sentence[start:end]
                method_list.append(method_data)
            elif sentence[num] == '^':
                num += 1
                start = num
                count = 0
                while (sentence[num] != '^'):
                    num += 1
                    count += 1
                end = start + count
                num += 1
                dataset_data = sentence[start:end]
                # print(type(dataset_data))
                dataset_list.append(dataset_data)
            else:
                num += 1
    return (method_list,dataset_list)

(method_list,dataset_list) = number_M_D(sens)
method_list.sort()
dataset_list.sort()
print('2800句话中方法实体的数量:',len(method_list),method_list[0:10]) 
print('2800句话中数据集实体的数量：',len(dataset_list)) 


"""句子重组"""
"""前1453组数据方法集和数据集同时更换""" #AAAI含有实体的句子为1453，循环一遍变成新的假的1453，加到2800中，现有2800+1453句话
#将MD_sentences（是sens） 替换后加到sens （是double_sens）上
def change(sens,lower_method_list,lower_dataset_list): #7899变成15798，将原先7899的实体随机换成小写，再加入到原先的7899句话，一共7899*2=15798句话
    final_row = len(sens)
    double_sens = []
    for row in range(final_row):
        sentence = sens[row]
        new_sentense = sentence
        num = 0
        while num < len(sentence):
            if sentence[num] == '~':
                num += 1
                start = num
                count = 0
                while (sentence[num] != '~'):
                    num += 1
                    count += 1
                end = start + count
                num += 1
                replace_method_data = sentence[start:end]
                new_sentense = new_sentense.replace(replace_method_data, choice(lower_method_list))
            elif sentence[num] == '^':
                num += 1
                start = num
                count = 0
                while (sentence[num] != '^'):
                    num += 1
                    count += 1
                end = start + count
                num += 1
                replace_dataset_data = sentence[start:end]
                new_sentense = new_sentense.replace(replace_dataset_data, choice(lower_dataset_list))
            else:
                num += 1
        double_sens.append(new_sentense)
    return double_sens

sens = change(MD_sentences,method_list,dataset_list)
#print(sens[0:3])
print('将增加数据集后的总句子进行替换一次后总句子数量',len(sens))  #2800+2800=5600 要考虑只加入含有实体的句子吗
sens2 = change(MD_sentences,method_list,dataset_list)
#print(sens2[0:3])
sens.extend(sens2)

sens3 = change(MD_sentences,method_list,dataset_list)
#print(sens3[0:2])
sens.extend(sens3) #重复三遍，保证大于2800，1300*3

sens4 = change(MD_sentences,method_list,dataset_list)
#print(sens4[0:2])
sens.extend(sens4) 

#sens5 = change(MD_sentences,method_list,dataset_list)
#print(sens5[0:2])
#sens.extend(sens5)

print(len(sens))
result_preserve = pd.DataFrame(sens,columns=['sentence']) #第1列为论文编号，第2列为论文名字，第3列为实验部分段落
result_preserve.to_excel("entity_replace-ACL-7000.xlsx",index = False) 
time2 = time.time()
print("总共消耗时间为:",time2-time1) 


"""对所有15798组句子进行读入和设置标签"""
def input_tag(sens): #生成不带标记的句子，和字母级别的标签
    sens_data = []
    tags_data = []
    for row in range(len(sens)):
        sentence = sens[row]
        num = 0
        sen_data = []
        tag_data = []
        while num < len(sentence):
            if sentence[num] == '~':
                num += 1
                sen_data += sentence[num]
                tag_data.append('B-M')
                num += 1
                while (sentence[num] != '~'):
                    #print(row,num)
                    sen_data += sentence[num]
                    tag_data.append('I-M')
                    num += 1
                num += 1
            elif sentence[num] == '^':
                num += 1
                sen_data += sentence[num]
                tag_data.append('B-D')
                num += 1
                while (sentence[num] != '^'):
                    sen_data += sentence[num]
                    tag_data.append('I-D')
                    num += 1
                num += 1
            else:
                sen_data += sentence[num]
                tag_data.append('O')
                num += 1
            # print(sen_data)
        sens_data.append(sen_data)
        tags_data.append(tag_data)
    return (sens_data,tags_data)


(sens_data, tags_data) = input_tag(sens)
print('扩增后不带标记的原始句子数量：',len(sens_data)) #15798
print('字母级别标注的句子数量:',len(tags_data))  #15798


"""读入规则标签"""
def rule_label(sens,data_dic,method_dic,none_dic):
    rules_label = []
    for row in range(len(sens)):
        sentence = sens[row]
        sentence = sentence.replace('~', '').replace('^', '').replace('`','')
        for data_str in data_dic:
            if sentence.find(dot+data_str+dot) != -1:
                sentence = sentence.replace(dot+data_str+dot, dot+add_d+data_str+add_d+dot)
        for method_str in method_dic:
            if sentence.find(dot+method_str+dot) != -1:
                sentence = sentence.replace(dot+method_str+dot, dot+add_m+method_str+add_m+dot)
        for none_str in none_dic:
            if sentence.find(dot+none_str+dot) != -1:
                sentence = sentence.replace(dot+none_str+dot, dot+add_n+none_str+add_n+dot)
        num = 0
        rule_label = []
        while num < len(sentence):
            if sentence[num] == '~':
                num += 1
                rule_label.append('B-M')
                num += 1
                while (sentence[num] != '~'):
                    rule_label.append('I-M')
                    num += 1
                num += 1
            elif sentence[num] == '^':
                num += 1
                rule_label.append('B-D')
                num += 1
                while (sentence[num] != '^'):
                    rule_label.append('I-D')
                    num += 1
                num += 1
            elif sentence[num] == '`':
                num += 1
                rule_label.append('O')
                num += 1
                while (sentence[num] != '`'):
                    rule_label.append('O')
                    num += 1
                num += 1
            else:
                rule_label.append('X')
                num += 1
            # print(sen_data)
        rules_label.append(rule_label)
    return rules_label
  
rules_label = rule_label(sens,data_dic,method_dic,none_dic)
print('规则标签的数目：',len(rules_label)) #15798
# # del sens_data[11]
# # del final_tags_data[11]
index = []
for row in range(len(sens_data)):
    if len(sens_data[row]) != len(rules_label[row]):
        index.append(row)
print('规则标签某个句子长度不等于该句子的长度：',index)
#print(sens_data[1311])
#print(rules_label[1311])


train_w, test_w, train_l, test_l, train_r, test_r = train_test_split(sens_data, tags_data, rules_label, test_size=0.1,
                                                                   random_state=40) #我想test_size=0.2,test_size=0.25，或0.2，0.125。叶原先为 0.1，0.15错了应该
train_w, dev_w, train_l, dev_l, train_r, dev_r= train_test_split(train_w, train_l, train_r, test_size=0.15, random_state=40)

print('训练集的输入句子:',len(train_w),'训练集的标签:',len(train_l), '训练集的规则的标签：',len(train_r)) #12085
print('测试集的输入原始句子：',len(test_w),'测试集的标签:',len(test_l), '测试集的规则的标签：',len(test_r)) #1580
print('交叉验证集的输入句子:',len(dev_w),'交叉验证集的标签：',len(dev_l), '交叉验证集规则的标签：',len(dev_r)) #2133


def bulid_csv(words, tags, filename):
    pd_dataframe = pd.DataFrame({'words': words[0], 'lable': tags[0]})
    zero_dataframe = pd.DataFrame({'words': [None], 'lable': [None]})
    #print(zero_dataframe)
    row = 1
    while row < len(words):
        pd_dataframe = pd_dataframe.append(zero_dataframe)
        #print(row)
        pd_add = pd.DataFrame({'words': words[row], 'lable': tags[row]})
        
        pd_dataframe = pd_dataframe.append(pd_add)
        row += 1
    pd_dataframe.to_csv(filename, index=False, header=False, sep=' ')

bulid_csv(dev_w, dev_l,"dev_AAAI.csv")
bulid_csv(train_w, train_l ,"train_AAAI.csv")
bulid_csv(test_w, test_l, "test_AAAI.csv")
bulid_csv(dev_w, dev_r,"dev1_AAAI.csv")
bulid_csv(train_w, train_r,"train1_AAAI.csv")
bulid_csv(test_w, test_r, "test1_AAAI.csv")
time2 = time.time()
print("总共消耗时间为:",time2-time1) # AAAI存四个表格需要7分钟



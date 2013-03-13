# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:39:53 2012

@author: qingfeng
"""

import os
import jieba
from gensim import corpora,parsing, models
import logging, gensim
import numpy as np


# 分词得到一个词语列表,unicode编码
# 分词包括去停用词

def fenci(str1): 
    seg_list = jieba.cut(str1,cut_all=False)
    list1=[]
    for ciyu in seg_list: # ciyu一定是unicode
        if len(ciyu)>1:
            list1.append(ciyu)

    
    list2=[] # list2是为了去掉list1中的多个回车

    for item in list1: # 去回车
        if '\n' in item:
            continue
        else:
            list2.append(item)
            
    # 去中文停用词
    with open('stopwords.dat','r') as fp:
        stopwords=fp.readlines() # utf8
    stopws=[]
    for item in stopwords:
        stopws.append(item.rstrip().decode('utf-8')) # 去掉最后的回车符,同时变成unicode
    
    
    list3=[]    
    for word in list2:
        if word not in stopws:
            list3.append(word)
    
    return list3



def getDictCorpus():
    # 构造corpus
    allFiles=os.listdir(os.getcwd())
    
    usefulFiles=[]
    for item in allFiles:
        if item[-3:] == 'txt':
            usefulFiles.append(item)
            
    contentList=[]
    for name in usefulFiles:
        fp=open(name,'r') 
        content=fp.read() # utf8
        c1=content.decode('utf-8') # c1 unicode
        
        list1=fenci(c1)
        
        if len(list1)!=0: # 此篇文章非空且不是乱码
            contentList.append(list1)
    
    dictionary = corpora.Dictionary(contentList)
    # 得到了字典
    dictionary.save('people.dict')
    
    corpus = [dictionary.doc2bow(text) for text in contentList]
    corpora.MmCorpus.serialize('people.mm', corpus)
    
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # 得到语料库
    corpora.MmCorpus.serialize('people_tfidf.mm',corpus_tfidf)
    
    



getDictCorpus()
    
    
    
    
    
    
    
    
    
    
    
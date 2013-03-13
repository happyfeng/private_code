# -*- coding: utf-8 -*-
"""
Created on Tue Dec 04 08:45:46 2012

@author: qingfeng
"""


from gensim import corpora,parsing, models
import logging, gensim
import wordcloud
import numpy as np

from preprocess import fenci

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 主题数目
TOPICS=20
# 单词云中单词的数目
WORDS=10
# 存储主题的单词概率分布
batchTopics=[]
onlineTopics=[]

# articleName是中文,OK
def batch_vs_online(articleName):
    # 载入字典和语料库
    diction = gensim.corpora.Dictionary.load('people.dict')
    mm = gensim.corpora.MmCorpus('people_tfidf.mm')
    # 读入新的文章
    with open(articleName,'r') as fp:
        content=fp.read()
        list1=fenci(content) # 停用词已经去掉
    
    doc_bow=diction.doc2bow(list1) # 统计词频后的向量,新文章的向量表示
    
    
    # 得到batch_lda与online_lda
    # alpha,eta使用默认参数
    batch_lda=gensim.models.ldamodel.LdaModel(corpus=mm, id2word=diction, \
    num_topics=TOPICS, update_every=0, passes=20)
    
    """
    batch_lda=gensim.models.ldamodel.LdaModel(corpus=mm, id2word=diction, \
    num_topics=TOPICS, update_every=0, passes=80,alpha=50.0/TOPICS)
    """
   
    
    # 主题数大一点，chunksize大一点，alpha要自己设置
    online_lda=gensim.models.ldamodel.LdaModel(corpus=mm, id2word=diction, \
    num_topics=TOPICS, update_every=1, chunksize=5, passes=1)
    
    """
    online_lda=gensim.models.ldamodel.LdaModel(corpus=mm, id2word=diction, \
    num_topics=TOPICS, update_every=1, chunksize=40, passes=80,alpha=50.0/TOPICS)
    """
    
    # batchTopics=[[],[],[]]
    # 每个主题下，单词已经按概率大小排列好了
    # list1=lda.show_topic(0)
    # list1=[(0.0017132658909227052, '\xb7\xa8\xd4\xba'), (0.0016304890553909524, '\xd4\xbd\xc0\xb4\xd4\xbd')]
    for k in range(TOPICS):
        
        batchTopics.append(batch_lda.show_topic(k))
        onlineTopics.append(online_lda.show_topic(k))
    
    
    # 先画batch_lda的单词云
    # 一篇文档中，各个主题所占的比重，是一个列表
    
    doc_batch_lda = batch_lda[doc_bow]
    
    print doc_batch_lda
    
    
    
    tP_batch=[]
    for yuanzu in doc_batch_lda:
        tP_batch.append(list(yuanzu))
    
    for i in range(len(tP_batch)):
        tmp=tP_batch[i][0]
        tP_batch[i][0]=tP_batch[i][1]
        tP_batch[i][1]=tmp
    
    # 从小到大排序
    tP_batch.sort()
    # 取3个主题   [主题所占'比重'，主题]
    tP_batch_new=tP_batch[-3:]
    #print tP_batch
    # 归一标准化
    sum0=0
    for i in range(len(tP_batch_new)):
        sum0+=tP_batch_new[i][0]
    
    for i in range(len(tP_batch_new)):
        tP_batch_new[i][0]=tP_batch_new[i][0]/sum0
    # now, tP_new=[[0.1,8],[0.3,2],[0.6,1]]
    
    
    
    # 画单词云用
    batchWordsList=[]
    batchWordsCount=[]
    
    for (rate,topic) in tP_batch_new:
        if rate>=0.1:
            # 四舍五入
            wordsNum=int(round(WORDS*rate))
            # 抽取主题topic，wordsNum个单词，加入words和counts
            for i in range(wordsNum):
                batchWordsList.append(batchTopics[topic][i][1])
                batchWordsCount.append(batchTopics[topic][i][0])

    
    

              
    
    words=np.array(batchWordsList)
    counts=np.array(batchWordsCount)

    font_path=r'C:\Windows\Fonts\simsun.ttc' # 宋体
    imageName='batch_lda_'+articleName
    wordcloud.make_wordcloud(words,counts,font_path,imageName)
    
    
    # online_lda的单词云
    doc_online_lda = online_lda[doc_bow]
    print doc_online_lda
    
    
    tP_online=[]
    for yuanzu in doc_online_lda:
        tP_online.append(list(yuanzu))
    
    for i in range(len(tP_online)):
        tmp=tP_online[i][0]
        tP_online[i][0]=tP_online[i][1]
        tP_online[i][1]=tmp
    
    # 从小到大排序
    tP_online.sort()
    # 取3个主题   [主题所占'比重'，主题]
    tP_online_new=tP_online[-3:]
    
    # 归一标准化
    sum0=0
    for i in range(len(tP_online_new)):
        sum0+=tP_online_new[i][0]
    
    for i in range(len(tP_online_new)):
        tP_online_new[i][0]=tP_online_new[i][0]/sum0
    # now, tP_new=[[0.1,8],[0.3,2],[0.6,1]]
                                
            
    onlineWordsList=[]
    onlineWordsCount=[]
    
    for (rate,topic) in tP_online_new:
        if rate>=0.1:
            # 四舍五入
            wordsNum=int(round(WORDS*rate))
            # 抽取主题topic，wordsNum个单词，加入words和counts
            for i in range(wordsNum):
                onlineWordsList.append(onlineTopics[topic][i][1])
                onlineWordsCount.append(onlineTopics[topic][i][0])
                


    
    words=np.array(onlineWordsList)
    counts=np.array(onlineWordsCount)
    font_path=r'C:\Windows\Fonts\simsun.ttc' # 宋体
    imageName='online_lda_'+articleName
    wordcloud.make_wordcloud(words,counts,font_path,imageName)
    
    
    
    
    
    
batch_vs_online('nvpai.txt')

    
    
    
    
    
    
    



# -*- coding: utf-8 -*-


"""
# 构造training.dat和vocab.dat，holdout.dat,小明已完成
# ubuntu下运行得到final-log-beta.dat（主题单词概率），holdout-phi-sum.dat（新文档主题比重）


"""

import wordcloud
import numpy as np
from ctm_topics import save_topics


# 主题数目
TOPICS=20
# 单词云中单词的数目
WORDS=10


def ctm_word_cloud():
    # 新文档中各个主题所占的比重，只取前两个主题
    with open('holdout-phi-sum.dat','r') as fp:
        topicProportion=fp.readlines()
        # 列表中的数字是float型
        topicProportion=map(lambda x: float(x.split()[0]),topicProportion)
        # tP=[[10,0],[90,1],[50,2],..]
        tP=[]
        for i in range(len(topicProportion)):
            item1=list((topicProportion[i],i))
            tP.append(item1)

        # 从小到大排序
        tP.sort()
        # 取两个主题,tP_new=[[75,8],[90,2]],    [主题所占'比重'，主题]
        tP_new=tP[-2:]
        
        sum2=tP_new[0][0]+tP_new[1][0]
        
        for i in range(2):
            tP_new[i][0]=tP_new[i][0]/sum2
        
        # now, tP_new=[[0.4,8],[0.6,2]]
        
    
    
    
    topics=save_topics('final-log-beta.dat', 'vocab.dat')
    
    # 构造ctm单词云
    wordsList=[]
    countsList=[]
    for (rate,topic) in tP_new:
        if rate>=0.1:
            # 四舍五入
            wordsNum=int(round(WORDS*rate))
            # 抽取主题topic，wordsNum个单词，加入words和counts
            for i in range(wordsNum):
                wordsList.append(topics[topic][i][1])
                countsList.append(topics[topic][i][0])
    
    words=np.array(wordsList)
    counts=np.array(countsList)
    font_path=r'C:\Windows\Fonts\simsun.ttc' # 宋体
    imageName='ctm'
    wordcloud.make_wordcloud(words,counts,font_path,imageName)
    



#  
ctm_word_cloud()                
                
                
                
        
    
    
        
        
        
        



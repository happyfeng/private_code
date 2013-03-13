#! /usr/bin/python

# usage: python topics.py <beta-file> <vocab-file> <num words>

import sys, numpy

def save_topics(beta_file, vocab_file,
                 nwords = 10):

    # get the vocabulary
    myTopics=[]
	
    vocab = file(vocab_file, 'r').readlines()
	# remove '\n'
    vocab = map(lambda x: x.strip(), vocab)
    vocab= map(lambda x: x.decode('gb2312'), vocab)

    indices = range(len(vocab)) # a list
	# (1,104730) array
	# original, it is a str type
    topic = numpy.array(map(float, file(beta_file, 'r').readlines()))

    nterms  = len(vocab)
    ntopics = len(topic)/nterms
    topic   = numpy.reshape(topic, [ntopics, nterms])
    for i in range(ntopics):
		# myTopic=[(probability,word),...]
        myTopic=[]
        indices.sort(lambda x,y: -cmp(topic[i,x], topic[i,y]))
		# log p
        for j in range(nwords):
            # out.write('     '+vocab[indices[j]]+'  '+str(2**(topic[i][indices[j]]))+'\n')
		myTopic.append((2**(topic[i][indices[j]]),vocab[indices[j]]))
        myTopics.append(myTopic)
		
    return myTopics


if (__name__ == '__main__'):
     beta_file = sys.argv[1]
     vocab_file = sys.argv[2]
     nwords = int(sys.argv[3])
     print_topics(beta_file, vocab_file, nwords)

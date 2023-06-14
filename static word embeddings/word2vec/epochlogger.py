import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import RegexpTokenizer
import gensim
import re
import multiprocessing
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import torch
import time
import datetime
from gensim.models.callbacks import CallbackAny2Vec

file = open( "weat.txt", "r" ) 
file=file.read()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(file.lower())

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 1

    def print_time(self):
        now = datetime.datetime.now()
        milliseconds = int(now.microsecond / 1000)  # Get the milliseconds component
        print(now.strftime("%Y-%m-%d %H:%M:%S") + f".{milliseconds:03d}")
        print('\n')
#         system_time = time.time()
#         readable_time = time.ctime(system_time)
#         print(readable_time)
    
    def target_attribute(self,target_one,target_two, target_one_words, attribute_one,attribute_two, attribute_one_words, target_two_words, attribute_two_words, model):
        cos=[]
        s=0
        s1=[]
        s2=[]
        S=[]
        n=0
        for i in range(0, len(target_one_words)):
          c1=[]
          c2=[]
          for k in range(0, len(attribute_one_words)):
            wt = target_one_words[i][:-1]
            at1 = attribute_one_words[k][:-1]
            try:
              cos1= model.wv.similarity(wt, at1)
              cos.append(cos1)
              c1.append(cos1)
            except:
              cos1=0
              cos.append(cos1)
              c1.append(cos1)
              continue
          for k in range(0, len(attribute_two_words)):
            cos2=0
            wt = target_one_words[i][:-1]
            at2 = attribute_two_words[k][:-1]
            try:
              cos2= model.wv.similarity(wt, at2)
              cos.append(cos2)
              c2.append(cos2)
            except:
              cos2=0
              cos.append(cos2)
              c2.append(cos2)
              continue
          s1.append((np.mean(c1)-np.mean(c2)))
          S.append((np.mean(c1)-np.mean(c2)))
          n=n+1

        for i in range(0, len(target_two_words)):
          c1=[]
          c2=[]

          for k in range(0, len(attribute_one_words)):
            wt = target_two_words[i][:-1]
            at1 = attribute_one_words[k][:-1]
            try:
              cos1= model.wv.similarity(wt, at1)
              cos.append(cos1)
              c1.append(cos1)
            except:
              cos1=0
              cos.append(cos1)
              c1.append(cos1)
              continue

          for k in range(0, len(attribute_two_words)):
            cos2=0
            wt = target_two_words[i][:-1]
            at2 = attribute_two_words[k][:-1]
            try:
              cos2= model.wv.similarity(wt, at2)
              cos.append(cos2)
              c2.append(cos2)
            except:
              cos2=0
              cos.append(cos2)
              c2.append(cos2)
              continue
          s2.append((np.mean(c1)-np.mean(c2)))
          S.append((np.mean(c1)-np.mean(c2)))
        s=np.sum(s1)-np.sum(s2)
        stdev=np.std(S)
        print(target_one + ' vs ' + target_two  + ' , ' +attribute_one + ' vs ' + attribute_two +', d = ' + str(s/(stdev*n)))
    
    def weat_test(self, model):
        for i in range(len(raw_sentences)-30):
            words=raw_sentences[i*4].split()
            target_one = words[0][:-1]
            target_one_words = words[1:]
            words1=raw_sentences[i*4+1].split()
            target_two = words1[0][:-1]
            target_two_words = words1[1:]
            words2=raw_sentences[i*4+2].split()
            attribute_one = words2[0][:-1]
            attribute_one_words = words2[1:]
            words3 = raw_sentences[i*4+3].split()
            attribute_two = words3[0][:-1]
            attribute_two_words = words3[1:]
            self.target_attribute(target_one,target_two, target_one_words, attribute_one,
                            attribute_two, attribute_one_words, target_two_words, attribute_two_words, model)
   
    def on_epoch_begin(self, model):
        print("Epoch #{} start time is:".format(self.epoch))
        self.print_time()


    def on_epoch_end(self, model):
        print("Epoch #{} end time is:\n".format(self.epoch))
        self.print_time()
        
        if(self.epoch % 2 == 0):
            print('\n')
            self.weat_test(model)
        
        loss = model.get_latest_training_loss()
        print('\nLoss after epoch {}: {}\n'.format(self.epoch, loss))
        self.epoch += 1






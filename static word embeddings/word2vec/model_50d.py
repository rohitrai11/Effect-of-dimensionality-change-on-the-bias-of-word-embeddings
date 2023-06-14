from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from gensim.models.callbacks import CallbackAny2Vec
from epochlogger import EpochLogger
from gensim.test.utils import common_texts
import multiprocessing
import torch
import time

workers=multiprocessing.cpu_count()
print(workers)
print('\n')


sentences = []
# open file in read mode
# give full path of preprocessed wikipedia dump file
f = open('sent3.txt', 'r')

# get the system time
system_time = time.time()

# convert system time to readable format
readable_time = time.ctime(system_time)

# print the system time
print("\nSystem time before preparing dataset is:\n", readable_time)

# display content of the file
for x in f.readlines():
  l = x.split(',')
  #print(len(l[:-1]))
  sentences.append(l[:-1])
# close the file
f.close()

# get the system time
system_time = time.time()

# convert system time to readable format
readable_time = time.ctime(system_time)

# print the system time
print("\nSystem time after preparing dataset is:\n", readable_time)

print(len(sentences))
print('\n')

epoch_logger = EpochLogger()


model_50d = Word2Vec(alpha=0.05,
                  vector_size=50,
                  window=5,
                  workers=multiprocessing.cpu_count(),
                  min_count=10)


# get the system time
system_time = time.time()

# convert system time to readable format
readable_time = time.ctime(system_time)

# print the system time
print("\nSystem time before building vocab is:\n", readable_time)

model_50d.build_vocab(sentences, progress_per=1000000)

# get the system time
system_time = time.time()

# convert system time to readable format
readable_time = time.ctime(system_time)

# print the system time
print("\nSystem time after building vocab is:\n", readable_time)

model_50d.train(sentences, total_examples=model_50d.corpus_count, epochs=20, compute_loss=True, callbacks=[epoch_logger])

# specify the path to save the model
model_50d.save("/word2vec/model_50d/word2vec_50d.model")

m = Word2Vec.load("/word2vec/model_50d/word2vec_50d.model")

print(m.wv.vectors.shape)

# -*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd

#s = open('msr_training.utf8').read()
s = open('msr_training.txt', 'r', encoding='cp936').read()
s = s.split('\r\n')

word_size = 128
maxlen = 32
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model

def clean(s): #整理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

s = u''.join(list(map(clean, s)))

s = re.split(u'[，。！？、]', s)

data = [] #生成训练样本
label = []
def get_xy(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:,0]), list(s[:,1])

for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        label.append(x[1])

d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
d = d[d['data'].apply(len) <= maxlen]
d.index = range(len(d))
tag = pd.Series({'s':0, 'b':1, 'm':2, 'e':3, 'x':4})

chars = [] #统计所有字，跟每个字编号
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars)+1)

#生成适合模型输入的格式
from keras.utils import np_utils
d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x))))
d['y'] = d['label'].apply(lambda x: np.array(list(map(lambda y:np_utils.to_categorical(y,5), tag[x].values.reshape((-1,1))))+[np.array([[0,0,0,0,1]])]*(maxlen-len(x))))

#设计模型
# word_size = 128
# maxlen = 32
# from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
# from keras.models import Model

sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(len(chars)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
model = Model(inputs=sequence, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 1024
history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), batch_size=batch_size, epochs=20)

#转移概率，单纯用了等概率
zy = {'be':0.5, 
      'bm':0.5, 
      'eb':0.5, 
      'es':0.5, 
      'me':0.5, 
      'mm':0.5,
      'sb':0.5, 
      'ss':0.5
     }

zy = {i:np.log(zy[i]) for i in zy.keys()}

def viterbi(nodes):
    paths = {'b':nodes[0]['b'], 's':nodes[0]['s']}
    print(nodes)
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(nows.values())
            print(nows.values())
            paths[list(nows.keys())[k]] = list(nows.values())[k]
    return list(paths.keys())[np.argmax(list(paths.values()))]

def simple_cut(s):
    if s:
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int))+[0]*(maxlen-len(s))]), verbose=False)[0][:len(s)]
        r = np.log(r)
        nodes = [dict(zip(['s','b','m','e'], i[:4])) for i in r]
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    else:
        return []

not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')
def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result

print(model.summary())

texts = '''结婚的和尚未结婚的。苏剑林是科学空间的博主。广东省云浮市新兴县魏则西是一名大学生。这真是不堪入目的环境。列夫·托尔斯泰'''

res = cut_word(texts)
print(res)
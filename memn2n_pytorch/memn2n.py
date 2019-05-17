# @Time    : 2019/5/17 14:22
# @Author  : shakespere
# @FileName: memn2n.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from utils import to_var
class MemNN(nn.Module):
    def __init__(self,vocab_size,embed_size,max_story_len,hops=3,dropout=0.2,te=True,pe=True):
        super(MemNN, self).__init__()
        self.hops = hops
        self.embed_size = embed_size
        self.temporal_encoding = te
        self.position_encoding = pe
        self.max_story_len = max_story_len

        init_rng = 0.1
        self.dropout = nn.Dropout(p=dropout)
        self.A = nn.ModuleList([nn.Embedding(vocab_size,embed_size) for _ in range(hops+1)])
        for i in range(len(self.A)):
            self.A[i].weight.data.normal_(0,init_rng)
            self.A[i].weight.data[0] = 0 # for padding index
        self.B = self.A[0] #query encoder

        #Temporal encoding
        if self.temporal_encoding:
            self.TA = nn.Parameter(torch.Tensor(1,max_story_len+1,embed_size).normal_(0,0.1))
            self.TC = nn.Parameter(torch.Tensor(1,max_story_len+1,embed_size).normal_(0,0.1))
    def forward(self,x,q):
        '''
        :param x: [bs,story_len,s_sent_len]
        :param q: [bs,q_sent_len]
        :return:
        '''
        bs = x.size(0)

        story_len = x.size(1)
        s_sent_len = x.size(2)
        # if story_len >=self.max_story_len:
        #     story_len = self.max_story_len


        #position Encoding
        if self.position_encoding:
            J = s_sent_len
            d = self.embed_size
            pe = to_var(torch.zeros(J,d))#[s_sent_len,embed_size]
            for j in range(1,J+1):
                for k in range(1,d+1):
                    l_kj = (1 - j/J)-(k/d)*(1-2*j/J)
                    pe[j-1][k-1] = l_kj
            pe = pe.unsqueeze(0).unsqueeze(0)#[1,1,s_sent_len,embed_size]
            pe = pe.repeat(bs,story_len,1,1)#[bs,story_len,s_sent_len,embed_size]
        x = x.view(bs*story_len,-1) #[bs*story_len,s_sent_len]
        u = self.dropout(self.B(q)) #[bs,q_sent_len,embed_size]
        u = torch.sum(u,1)# (bs, embd_size)
        # Adjacent weight tying
        for k in range(self.hops):
            m = self.dropout(self.A[k](x)) #[bs*story_len,s_sent_len,embed_size]
            m = m.view(bs,story_len,s_sent_len,-1) #[bs,story_len,s_sent_len,embed_size]
            if self.position_encoding:
                m*=pe #[bs,story_len,s_sent_len,embed_size]
            m = torch.sum(m,2) #[bs,story_len,embed_size]
            m = m[:,:story_len,:]
            if self.temporal_encoding:
                # print(m.size(1))
                # if m.size(1)>self.TA.size(1):
                #     n = self.TA.repeat(bs,1,1)[1]

                n = self.TA.repeat(bs,1,1)[:,:story_len,:]
                # print(story_len)
                # print(n.size(1))
                # print('-----')
                m+=n

            c = self.dropout(self.A[k+1](x)) #[bs*story_len,s_sent_len,embed_size]
            c = c.view(bs,story_len,s_sent_len,-1)#[bs,story_len,s_sent_len,embed_size]
            c = torch.sum(c,2) # (bs, story_len, embd_size)
            if self.temporal_encoding:
                c+=self.TC.repeat(bs,1,1)[:,:story_len,:]# (bs, story_len, embd_size)

            p = torch.bmm(m,u.unsqueeze(2)).squeeze() #[bs,story_len]
            p = F.softmax(p,-1).unsqueeze(1) #[bs,1,story_len]
            o = torch.bmm(p,c).squeeze(1) #use m as c ,[bs,embed_size]
            u = o+u #[bs,embed_size]
        W = torch.t(self.A[-1].weight) #[embed_szie,vocab_size]
        out = torch.bmm(u.unsqueeze(1),W.unsqueeze(0).repeat(bs,1,1)).squeeze() #[bs,1,embed_size] * [bs,embed_size,vocab_size] ==> [bs,ansize,vocab_size] ==>[bs,ansize]
        return F.log_softmax(out,-1)



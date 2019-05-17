# @Time    : 2019/5/17 15:31
# @Author  : shakespere
# @FileName: train.py
import random
import os
import argparse
import torch
import torch.nn as nn
from utils import load_data,to_var,vectorize
from memn2n import MemNN

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
parser = argparse.ArgumentParser()
parser.add_argument('--embed_size',type=int,default=30,help='default 30.word embedding size')
parser.add_argument('--batch_size', type=int, default=256, help='default 32. input batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='resume epoch count, default=0')
parser.add_argument('--n_epochs', type=int, default=100, help='default 100. the number of epochs')
parser.add_argument('--max_story_len', type=int, default=25, help='default 25. max story length. see 4.2')
parser.add_argument('--use_10k', type=int, default=1, help='default 1. use 10k or 1k dataset')
parser.add_argument('--test', type=int, default=0, help='defalut 1. for test, or for training')
parser.add_argument('--resume', type=int, default=1, help='defalut 1. read pretrained models')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()
torch.manual_seed(args.seed)
PAD = '<PAD>'
max_story_len = args.max_story_len
embed_size = args.embed_size
batch_size = args.batch_size
n_epochs = args.n_epochs
use_10k  = args.use_10k

def save_checkpoint(state,is_best,filename):
    print("save model!",filename)
    torch.save(state,filename)
def custom_loss_fn(data,labels):
    loss = torch.autograd.Variable(torch.zeros(1))
    for d,label in zip(data,labels):
        loss-=torch.log(d[label]).cpu()
    loss/=data.size(0)
    return loss
def test(model,data,w2i,batch_size,task_id):
    model.eval()
    correct = 0
    count = 0
    for i in range(0,len(data)-batch_size,batch_size):
        batch_data = data[i:i+batch_size]
        story = [d[0] for d in batch_data]
        q = [d[1] for d in batch_data]
        a = [d[2][0] for d in batch_data]

        story_len = min(max_story_len,max([len(s) for s in story]))
        s_sent_len = max([len(sent) for s in story for sent in s])
        q_sent_len = max([len(sent) for sent in q])

        vec_data = vectorize(batch_data,w2i,story_len,s_sent_len,q_sent_len)
        story = [d[0] for d in vec_data]
        q = [d[1] for d in vec_data]
        a = [d[2][0] for d in vec_data]

        story = to_var(torch.LongTensor(story))
        q = to_var(torch.LongTensor(q))
        a = to_var(torch.LongTensor(a))
        pred = model(story,q)
        pred_idx = pred.max(1)[1]
        correct += torch.sum(pred_idx == a).item()
        count+=batch_size
    acc = correct/count*100
    print('Task {} Test Acc: {:.2f}% - '.format(task_id,acc),correct,'/',count)
    return acc
def adjust_lr(optimizer,epoch):
    if (epoch + 1)% 25 == 0:
        for pg in optimizer.param_groups:
            pg['lr'] *=0.5
            print('Learning rate is set to',pg['lr'])
def train(model,train_data,test_data,optimizer,loss_fn,w2i,task_id,batch_size,n_epoch):
    for epoch in range(n_epoch):
        model.train()
        correct = 0
        count =0
        random.shuffle(train_data)
        for i in range(0,len(train_data)-batch_size,batch_size):
            batch_data = train_data[i:i+batch_size]
            story = [d[0] for d in batch_data]
            story_len = min(max_story_len,max([len(s) for s in story]))
            s_sent_len = max([len(sent) for s in story for sent in s])
            q = [d[1] for d in batch_data]
            q_sent_len = max([len(sent) for sent in q])

            vec_data = vectorize(batch_data,w2i,story_len,s_sent_len,q_sent_len)
            story = [d[0] for d in vec_data]
            q = [d[1] for d in vec_data]
            a = [d[2][0] for d in vec_data]

            story = to_var(torch.LongTensor(story))
            q = to_var(torch.LongTensor(q))
            a = to_var(torch.LongTensor(a))

            pred = model(story,q)

            loss = loss_fn(pred,a)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            # reset padding index weight
            for name,param in model.named_parameters():
                if param.grad is not None:
                    if 'A.' in name:
                        param.data[0] = 0
            pred_idx = pred.max(1)[1]
            correct +=torch.sum(pred_idx==a).item()
            count+=batch_size
            # 梯度裁剪
            # for p in model.parameters():
            #     torch.nn.utils.clip_grad_norm(p, 40.0)
        if epoch % 20 ==0:
            print('==========Epoch {} =========='.format(epoch))
            print('Training Acc: {:.2f}% - '.format(correct/count*100),correct,'/',count)
            test(model,test_data,w2i,batch_size,task_id)
        # adjust_lr(optimizer,epoch)
def generate_model_filename(task_id,data_size,n_epochs):
    return '{}/Task_{}_{}-Epoch{}.model'.format('./checkpoints',data_size,task_id,n_epochs)
def run():
    test_acc_results = []
    for task_id in range(1,21):
        print("------------Task '{}'-----------".format(task_id))
        if use_10k:
            train_data,test_data,vocab = load_data('../data/tasks_1-20_v1-2/en-10k',0,task_id)
        else:
            train_data,test_data,vocab = load_data('../data/tasks_1-20_v1-2/en',0,task_id)
        data = train_data + test_data
        print('sample',train_data[0])
        w2i = dict((w,i) for i,w in enumerate(vocab,1))#小标从1开始
        w2i[PAD] = 0
        vocab_size = len(vocab)+1
        story_len = min(max_story_len,max(len(s) for s,q,a in data))
        s_sent_len = max(len(ss) for s,q,a in data for ss in s)
        q_sent_len = max(len(q) for s,q,a in data)
        print('train num', len(train_data))
        print('test num', len(test_data))
        print('vocab_size', vocab_size)
        print('embed_size', embed_size)
        print('story_len', story_len)
        print('s_sent_len', s_sent_len)
        print('q_sent_len', q_sent_len)
        model = MemNN(vocab_size,embed_size,vocab_size,story_len)
        model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.NLLLoss()
        ds = '10k' if use_10k else '1k'
        model_filename = generate_model_filename(task_id,ds,n_epochs)
        if os.path.isfile(model_filename) and args.resume:
            print("=> loading checkpoint '{}'".format(model_filename))
            checkpoint = torch.load(model_filename)
            args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print('no checkpoint found ad "{}"'.format(model_filename))
        if args.test!=1:
            train(model,train_data,test_data,optimizer,loss_fn,w2i,task_id,batch_size,n_epochs)
            save_checkpoint({
                'epoch':args.n_epochs,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict()
            },True,filename=model_filename)
            print('Final Acc')
            acc = test(model,test_data,w2i,batch_size,task_id)
            test_acc_results.append(acc)
        else:
            acc = test(model,test_data,w2i,batch_size,task_id)
            test_acc_results.append(acc)
    for i,acc in enumerate(test_acc_results):
        print('Task {}: Acc {:.2f}%'.format(i+1,acc))

if __name__ == '__main__':
    run()

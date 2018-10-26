import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, sentence_size, device):
        super(InputEncoder, self).__init__()
        #self.masks = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.mask = torch.FloatTensor(sentence_size, embed_dim).fill_(1).to(device)
        #self.masks._parameters['weight'].data.fill_(1)

    def forward(self, embedx):
        # approximate each sentence of the story to a single embedding
        return torch.sum(embedx * self.mask, 2)
        #maskx = self.masks(x)
        #return torch.sum(torch.mul(maskx, embedx), 2)

class DynamicMemory(nn.Module):
    def __init__(self, blocks, embed_size, batch_size, device):
        super(DynamicMemory, self).__init__()
        self.h = torch.FloatTensor(blocks, batch_size, embed_size).to(device)
        self.w = nn.Embedding(blocks, embed_size)
        self.U = nn.Linear(embed_size, embed_size, bias=False)
        self.V = nn.Linear(embed_size, embed_size, bias=False)
        self.W = nn.Linear(embed_size, embed_size, bias=False)
        self.bias = torch.FloatTensor(embed_size).fill_(0).to(device)
        self.prelu = nn.PReLU(num_parameters=embed_size, init=1.0)

        self.U.weight.data.normal_(0.0, 0.1)
        self.V.weight.data.normal_(0.0, 0.1)
        self.W.weight.data.normal_(0.0, 0.1)
        self.batch_size = batch_size

        self.block_idx = [torch.LongTensor([_]).to(device) for _ in range(blocks)]

    def forward(self, x, h):
        #x -> bs x es x 1
        #TODO vectorize along blocks
        h_list = []
        for i in range(h.size(0)):
            w_i = self.w(self.block_idx[i])#.repeat(self.batch_size, 1)
            self.gate = F.sigmoid(torch.sum(x * h[i], 1) + torch.sum(x *  w_i, 1))
            # gate: bs x es
            self.h_tilde = self.prelu(
                    self.U(h[i]) + 
                    self.V(w_i) + 
                    self.W(x) +
                    self.bias)
            # h_tilde: bs x es
            h_i = h[i] + self.gate.unsqueeze(1) * self.h_tilde
            h_list.append(h_i / (torch.sum(h_i, 1, keepdim=True) + 1e-8))
        return torch.stack(h_list)

    def initialize_hidden(self):
        return nn.init.constant_(self.h, 0)


class OutputModule(nn.Module):
    def __init__(self, embed_size, vocab_size, _):
        super(OutputModule, self).__init__()
        self.sm = nn.Softmax()
        self.R = nn.Linear(embed_size, vocab_size, bias=False)
        self.H = nn.Linear(embed_size, embed_size, bias=False)
        self.prelu = nn.PReLU()
        self.R.weight.data.normal_(0.0, 0.1)
        self.H.weight.data.normal_(0.0, 0.1)

    def forward(self, last_state, query):
        attention = torch.sum(last_state * query, 2)
        attention = attention - torch.max(attention, 1, keepdim=True)[0]
        attention = self.sm(attention)
        u = torch.sum(attention.unsqueeze(2) * last_state, 0)
        return self.R(self.prelu(query + self.H(u)))


class REN(nn.Module):
    def __init__(self, vocab_size, embed_size, blocks, verb_size, object_size, batch_size, device, sent_size):
        super(REN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed._parameters['weight'].data.normal_(0.0, 0.1)
        self.story_enc = InputEncoder(vocab_size, embed_size, sent_size, device)
        self.query_enc = InputEncoder(vocab_size, embed_size, sent_size, device)
        self.mem = DynamicMemory(blocks, embed_size, batch_size, device)
        self.output = OutputModule(embed_size, vocab_size, object_size)

    def forward(self, x, query):
        # x -> bs x story x token of sentences
        # q -> bs x tokens
        x = self.embed(x)
        x = self.story_enc(x)
        # x -> bs x seq x embed_size
        #return torch.sum(x, 1)
        h = self.mem.initialize_hidden()
        for i in range(x.size(1)):
            h = self.mem(x[:,i,:].squeeze(1), h)
            #break
        last_state = h 
        return self.output(last_state, self.query_enc(query.unsqueeze(1)).squeeze(1))


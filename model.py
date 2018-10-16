import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(InputEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.masks = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed._parameters['weight'].data.fill_(1)
        self.masks._parameters['weight'].data.fill_(1)

    def forward(self, x):
        #x -> bs x seq x 1(index)
        embedx = self.embed(x)
        maskx = self.masks(x)
        # add along embed_dim
        return torch.sum(torch.mul(maskx, embedx), 2)

class DynamicMemory(nn.Module):
    def __init__(self, blocks, embed_size, batch_size):
        super(DynamicMemory, self).__init__()
        self.h = torch.FloatTensor(blocks, batch_size, embed_size)
        self.w = torch.FloatTensor(blocks, batch_size, embed_size)
        self.U = nn.Linear(embed_size, embed_size, bias=False)
        self.V = nn.Linear(embed_size, embed_size, bias=False)
        self.W = nn.Linear(embed_size, embed_size, bias=False)
        self.bias = torch.FloatTensor(embed_size).fill_(0)
        self.prelu = nn.PReLU()

        self.U.weight.data.normal_(0.0, 0.1)
        self.V.weight.data.normal_(0.0, 0.1)
        self.W.weight.data.normal_(0.0, 0.1)
        self.prelu.weight.data.fill_(1)

    def forward(self, x, h):
        #x -> bs x es x 1
        #TODO vectorize along blocks
        h_list = []
        for i in range(h.size(0)):
            self.gate = F.sigmoid(
                    x * h[i] +
                    x * self.w[i]
                    )
            # gate: bs x es
            self.h_tilde = self.prelu(
                    self.U(h[i]) + 
                    self.V(self.w[i]) + 
                    self.W(x) +
                    self.bias)
            # h_tilde: bs x es
            h_i = h[i] + torch.mul(self.gate, self.h_tilde) 
            h_list.append(h_i / (torch.sum(h_i, 1, keepdim=True) + 1e-8))
        return torch.stack(h_list)

    def initialize_hidden(self):
        self.h = nn.init.constant_(self.h, 1)
        self.w = nn.init.constant_(self.w, 1)
        return self.h, self.w


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
    def __init__(self, vocab_size, embed_size, blocks, verb_size, object_size, batch_size):
        super(REN, self).__init__()
        self.inp_enc = InputEncoder(vocab_size, embed_size)
        self.mem = DynamicMemory(blocks, embed_size, batch_size)
        self.output = OutputModule(embed_size, vocab_size, object_size)

    def forward(self, x, query):
        # x -> bs x seq x tokens
        # q -> bs x tokens
        x = self.inp_enc(x)
        # x -> bs x seq x embed_size
        #return torch.sum(x, 1)
        h, w = self.mem.initialize_hidden()
        for i in range(x.size(1)):
            h = self.mem(x[:,i,:].squeeze(1), h)
            #break
        last_state = h 
        return self.output(last_state, self.inp_enc(query.unsqueeze(1)).squeeze(1))


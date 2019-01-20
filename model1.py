import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEncoder(nn.Module):
    def __init__(self, sentence_size, embed_size, device):
        super(InputEncoder, self).__init__()
        #self.mask = nn.Linear(sentence_size, embed_size, bias=False)
        self.mask = nn.Parameter(torch.FloatTensor(sentence_size, embed_size).fill_(1), requires_grad=True)#.to(device).requires_grad_()
        #self.mask._parameters['weight'].data.fill_(1)
    def forward(self, x):
        return torch.sum(x * self.mask, 2)

class thres(nn.Threshold):
    def __init__(self, inplace=False):
        super(thres, self).__init__(0., 1., inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


class MemCell(nn.Module):
    def __init__(self, num_blocks, embed_size, activation, device):
        super(MemCell, self).__init__()
        #self.keys = keys
        self.num_blocks = num_blocks
        self.activation = activation
        self.embed_size = embed_size

        self.U = nn.Linear(embed_size, embed_size, bias=False)
        self.V = nn.Linear(embed_size, embed_size, bias=False)
        self.W = nn.Linear(embed_size, embed_size, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(embed_size).normal_(0.0, 0.1), requires_grad=True)#.to(device).requires_grad_()
        self.U.weight.data.normal_(0.0, 0.1)
        self.V.weight.data.normal_(0.0, 0.1)
        self.W.weight.data.normal_(0.0, 0.1)
        self.th = thres()

    def get_gate(self, state_j, key_j, inputs):
        a = torch.sum(inputs * state_j, dim=1)
        b = torch.sum(inputs * key_j, dim=1)
        return F.sigmoid(a + b)

    def get_candidate(self, state_j, key_j, inputs):
        key_V = self.V(key_j)
        state_U = self.U(state_j) + self.bias
        inputs_W = self.W(inputs)
        #key_V = torch.matmul(key_j, self.V)
        #state_U = torch.matmul(state_j, self.U) + self.bias
        #inputs_W = torch.matmul(inputs, self.W)
        return self.activation(state_U + inputs_W + key_V)

    def forward(self, x, state):
        state = torch.split(state, self.embed_size, 1)
        next_states = []
        for j, state_j in enumerate(state):
            key_j = self.keys[j].unsqueeze(0)
            gate_j = self.get_gate(state_j, key_j, x)
            candidate_j = self.get_candidate(state_j, key_j, x)

            state_j_next = state_j + gate_j.unsqueeze(-1) * candidate_j
            state_j_next_norm = torch.abs(torch.norm(state_j_next, p=2, dim=-1, keepdim=True)) + 1e-8

            # mask=torch.zeros(state_j_next.shape)
            # mask[state_j_next.nonzero()]=1
            # state_j_next[state_j_next<=0.0] = 1.0

            state_j_next = self.th(state_j_next) / state_j_next_norm

            next_states.append(state_j_next)
        state_next = torch.cat(next_states, dim=1)
        return state_next

    def zero_state(self, bs):
        zero_state = torch.cat([key.unsqueeze(0) for key in self.keys], 1)
        zero_state_batch = zero_state.repeat(bs, 1)
        return zero_state_batch

class OutputModule(nn.Module):
    def __init__(self, num_blocks, vocab_size, embed_size, activation, device):
        super(OutputModule, self).__init__()
        self.activation = activation
        self.num_blocks = num_blocks
        self.embed_size = embed_size
        self.R = nn.Linear(embed_size, vocab_size, bias=False)
        self.H = nn.Linear(embed_size, embed_size, bias=False)
        self.R.weight.data.normal_(0.0, 0.1)
        self.H.weight.data.normal_(0.0, 0.1)

    def forward(self, x, state):
        state = torch.stack(torch.split(state, self.embed_size, dim=1), dim=1)
        #print(state.shape)
        #print(x.shape)
        attention = torch.sum(state * x, dim=2)
        attention = attention - torch.max(attention, dim=-1, keepdim=True)[0] 
        attention = F.softmax(attention).unsqueeze(2)

        u = torch.sum(state * attention, dim=1)
        q = x.squeeze(1)
        y = self.R(self.activation(q + self.H(u)))
        return y

class REN1(nn.Module):
    def __init__(self, num_blocks, vocab_size, embed_size, device, sentence_size, query_size):
        super(REN1, self).__init__()
        vocab_size = vocab_size + num_blocks
        self.device = device
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.embedlayer = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embedlayer._parameters['weight'].data.normal_(0.0, 0.1)

        self.prelu = nn.PReLU(num_parameters=embed_size, init=1.0)
        self.story_enc = InputEncoder(sentence_size, embed_size, device)
        self.query_enc = InputEncoder(query_size, embed_size, device)

        self.cell = MemCell(num_blocks, embed_size, self.prelu, device)
        self.output = OutputModule(num_blocks, vocab_size, embed_size, self.prelu, device)

    def init_keys(self):
        keys = [torch.LongTensor([key]).to(self.device) for key in range(self.vocab_size - self.num_blocks, self.vocab_size)]
        keys = [self.embedlayer(key).squeeze(0) for key in keys]
        self.cell.keys = keys

    def forward(self, story, query):
        story_embedded = self.embedlayer(story)
        query_embedded = self.embedlayer(query.unsqueeze(1))
        story_embedded = self.story_enc(story_embedded)
        query_embedded = self.query_enc(query_embedded)
        initial_state = self.cell.zero_state(story.shape[0])
        for i in range(story_embedded.shape[1]):
            initial_state = self.cell(story_embedded[:,i,:], initial_state)
        outputs = self.output(query_embedded, initial_state)
        return outputs

import torch
import torch.nn as nn
from torch.nn import functional as F
from midi_transform import note_dim, velocity_dim, duration_dim, dtime_dim, output_seq_to_input_seq, input_seq_to_output_seq



def to_tensors(X, device, dtype=torch.float32):
    X_t = [torch.tensor(v,dtype=dtype,device=device) for v in X]
    return X_t


def to_seq(t):
  a = t.detach().cpu().numpy()
  return a


def gen_batch(X,Y,batch_size,device):
    indices = torch.randint(len(X),[batch_size])
    indices = list(map(int,indices))
    indices.sort(key=lambda x: -X[x].shape[0])
    batch_X = []
    batch_Y = []
    lengths = []
    for i in indices:
        batch_X.append(X[i])
        batch_Y.append(Y[i])
        lengths.append(X[i].shape[0]-1)
    batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True)
    batch_Y = torch.nn.utils.rnn.pad_sequence(batch_Y, batch_first=True, padding_value=-1)
    return batch_X[:,:-1,:], batch_Y[:,1:,:], lengths


class musicLSTM(nn.Module):
    def __init__(self, num_units=256, num_layers=2, num_output=(note_dim+velocity_dim+duration_dim+dtime_dim), device=torch.device("cuda")):
        super(self.__class__, self).__init__()
        self.emb = nn.Embedding(note_dim, 4, padding_idx=None)
        self.lstm = nn.LSTM(input_size=(4+3),hidden_size=num_units,num_layers=num_layers,batch_first=True)
        self.linear1 = nn.Linear(num_units,num_units)
        self.linear2 = nn.Linear(num_units,num_output)
        self.device = device
        self.num_units = num_units
        self.num_layers = num_layers
        self.hidden = None
        self.to(device)

    def forward(self,x,lengths):
        #pack
        x = torch.cat((self.emb(x[:,:,0].to(torch.long)),x[:,:,1:]),-1)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, batch_first=True, lengths=lengths)
        h, self.hidden = self.lstm(x, self.hidden)
        #unpack
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h,batch_first=True)
        # batch x max_seq_len x vec_dim
        h = self.linear1(h)
        h = F.dropout(h,0.5)
        h = F.relu(h)
        y = self.linear2(h)
        return y

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers,batch_size, self.num_units, dtype=torch.float32, device=self.device,requires_grad=True)
        c0 = torch.zeros(self.num_layers,batch_size, self.num_units, dtype=torch.float32, device=self.device,requires_grad=True)
        return  (h0,c0)


def loss(y_h,y):
    # mask = (x!=0).any(dim=-1,keepdim=True).to(torch.float32)
    CrossEntropy = nn.CrossEntropyLoss(ignore_index=-1)
    l0 = CrossEntropy(y_h[...,0:note_dim].transpose(1,2),y[...,0])
    l1 = CrossEntropy(y_h[...,note_dim:(note_dim+velocity_dim)].transpose(1,2),y[...,1])
    l2 = CrossEntropy(y_h[...,(note_dim+velocity_dim):(note_dim+velocity_dim+duration_dim)].transpose(1,2),y[...,2])
    l3 = CrossEntropy(y_h[...,(note_dim+velocity_dim+duration_dim):].transpose(1,2),y[...,3])
    return l0+l1+l2+l3


def train(model, X_train,Y_train, batch_size, num_iter, opt, scheduler=None):
    best_model = model.state_dict()
    best_loss = 999999
    model.train()
    for epoch in range(num_iter):
        model.zero_grad()
        x,y,lengths = gen_batch(X_train,Y_train,batch_size,model.device)
        model.hidden = model.init_hidden(x.shape[0])
        y_h = model(x,lengths)
        l = loss(y_h,y)
        if epoch % 10 == 0:
            print("iter: " + str(epoch) + "   loss: " + str(l.detach().cpu().numpy()))
        if l < best_loss:
            best_loss = l
        l.backward()
        opt.step()
        if scheduler != None:
            scheduler.step()
    model.load_state_dict(best_model)


def get_classes(y_h,random, temperature):
    if random:
        y0 = torch.multinomial(torch.softmax(y_h[0:note_dim] / temperature,-1), 1, replacement=True)
        y1 = torch.multinomial(torch.softmax(y_h[note_dim:(note_dim+velocity_dim)]/ temperature,-1), 1, replacement=True)
        y2 = torch.multinomial(torch.softmax(y_h[(note_dim+velocity_dim):(note_dim+velocity_dim+duration_dim)]/ temperature,-1), 1, replacement=True)
        y3 = torch.multinomial(torch.softmax(y_h[(note_dim+velocity_dim+duration_dim):]/ temperature,-1), 1, replacement=True)
    else:
        y0 = torch.argmax(y_h[0:note_dim],-1,keepdim=True)
        y1 = torch.argmax(y_h[note_dim:(note_dim+velocity_dim)],-1,keepdim=True)
        y2 = torch.argmax(y_h[(note_dim+velocity_dim):(note_dim+velocity_dim+duration_dim)],-1,keepdim=True)
        y3 = torch.argmax(y_h[(note_dim+velocity_dim+duration_dim):],-1,keepdim=True)
    return torch.cat((y0, y1, y2, y3), -1)


def generate(model, seed, length=200, random=False, temperature=1.0):
    model.eval()
    model.hidden = model.init_hidden(1)
    # seq = input_seq_to_output_seq(seed)
    seq = []
    with torch.no_grad():
        input_seed = torch.tensor(seed, device=model.device).unsqueeze(0)
        out = model.forward(input_seed,[len(seed)])
        seq.append(to_seq(get_classes(out[0,-1,:],random, temperature)))
        for i in range(length-1):
            input_tensor = torch.tensor(output_seq_to_input_seq([seq[i]]),dtype=torch.float32,device=model.device).unsqueeze(0)
            out = model.forward(input_tensor, [1])
            seq.append(to_seq(get_classes(out[0,-1,:],random,temperature)))
    return seq

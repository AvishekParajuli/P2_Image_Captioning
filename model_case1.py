import torch
import torch.nn as nn
import torchvision.models as models

#case1: numlayers =2, dropout=0.2 in lstm; train bn parameters; Adam(with L2 regularization]; xavier initialization
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum= 0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        #super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        #define word_embeddeing layer
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        #define LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)
        #initilize hidden layer
        self.init_weights()
    
    def forward(self, features, captions):
        # captions.shape: torch.Size([10, 11]), where v=11 is variable length of captions, and 10 corresponds to batch_size
        # features: output of embedding containg encoder output: shape: [10, 256]=[batch_size, embedding_size]
        batch_size = captions.shape[0]
        #at the start initilize hidden layer
        #self.hidden = self.init_hidden(batch_size)
        #print("captions.shape =", captions.shape)
        #print("caption shape without <end> = ", captions[:,:-1].shape)
        
        #Generate word_embedding vector for all except<end> token
        word_embeds = self.word_embed(captions[:,:-1]) 
        #word_embeds.shape =[10,V-1,embed_size]; batch_size=10; V= variable caption length
        #why V-1?
        #input sequence = ['<start>', 'Giraffes', 'standing', 'next', 'to', 'each', 'other']
        #target sequence = [ 'Giraffes', 'standing', 'next', 'to', 'each', 'other', <end>] 
        
        
        #features.shape =[batch_size, embed_size]=[10,256]
        #transform it into tensor of shape[10, 1, 256] similar to word_embeds
        features_pr = features.unsqueeze(1) 
        # concatenate along dimension 1 [10,1,256] + [10,V,256]
        #input = a Tensor containing the values in an input sequence; this has values: (seq_len, batch, input_size)
        inputs = torch.cat((features_pr, word_embeds),1)
        #shape of inputs is [10, V+1, 256] => lstm defined as batch_first=True for compatibility
        #print("shape of input to lstm", inputs.shape)
        # transform shape to [10, v,256]
        lstm_out, hidden = self.lstm(inputs)#input all except the last one
        #print("lstm_out.shape", lstm_out.shape)
        #print("hidden.shape", hidden.shape)
        tag_outputs = self.fc(lstm_out)
        # use softmax to get scores; if not using softmax use crossentropy in loss
        #tag_scores = F.log_softmax(tag_outputs, dim=1)
        return tag_outputs
    
    
    def init_weights(self):
        """Initialize weights."""
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        #self.fc.weight.data.uniform_(-0.1, 0.1)
        torch.nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.fill_(0.1)

    def sample(self, inputs, states=None, max_len=20):
        ''' accepts pre-processed image tensor (inputs) 
        and returns predicted sentence (list of tensor ids of length max_len) '''
        #print("inputs.shape = ", inputs.shape)
        #inputs: [bs=10,1,256]=[batch_size, 1, 256]
        #there are max_len=20 number of lstm; pass inputs sequentially to lstm and collect <words>
        caption_idxs =[]
        for i in range(max_len):
            out,states = self.lstm(inputs, states)     #out: [bs, 1, hidden_size]
            #taken hidden= lstm_out and pass through fc layer
            tag_outputs = self.fc(out)         #tag_outputs: [bs,1, vocab_size]
            tag_outputs = tag_outputs.squeeze(dim=1)  #tag_outputs: [ bs, vocab_size]
            #tag_outputs = logit_scores=> find the index(along dim =1) which has maximum value
            word_idx = tag_outputs.argmax(dim =1)  # word_idx =[1]
            caption_idxs.append(word_idx.item())
            
            #prepare input for the next sequence
            next_input = word_idx.unsqueeze(1)   #next_input: [1,1]
            inputs = self.word_embed(next_input)  #inputs: [1, 1,256]
            #print("next_input.shape ", inputs.shape)
        
        return caption_idxs
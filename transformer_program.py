def data_loader(x_data_tr, y_data_tr, x_data_val, y_data_val):
    xtr = torch.tensor(x_data_tr, dtype=torch.float32)
    ytr = torch.tensor(y_data_tr, dtype=torch.float32)
    print("len of xtr: {}".format(len(xtr)))
    print("len of ytr: {}".format(len(ytr)))
    train_dataset = torch.utils.data.TensorDataset(xtr, ytr)
    print("train_dataset"+"#"*20)
    print(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=32,#batch_size=64, batch_size=512,  4 is best.1. 2.
                                            shuffle=False)
    xval = torch.tensor(x_data_val, dtype=torch.float32)
    yval = torch.tensor(y_data_val, dtype=torch.float32)
    print("len of xval: {}".format(len(xval)))
    print("len of yval: {}".format(len(yval)))
    #yte = torch.tensor(y_test, dtype=torch.int64)
    val_dataset = torch.utils.data.TensorDataset(xval, yval)
    #テスト用 Dataloder
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=64,#batch_size=64, batch_size=512. 1. 8.
                                            shuffle=False)
    #test_loader = test_dataloader
    print("data loading has completed!")
    return train_dataloader, val_dataloader 

def test_data_loader(x_data_test, y_data_test):
    xtest = torch.tensor(x_data_test, dtype=torch.float32)
    ytest = torch.tensor(y_data_test, dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(xtest, ytest)
    #テスト用 Dataloder
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=1,#batch_size=64, batch_size=512. 1. 8.
                                            shuffle=False)
    #test_loader = test_dataloader
    print("data loading has completed!")
    return test_dataloader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        #pe_p = self.pe[:, :x.size(1)]
        pe_p = self.pe[:, :x.size(0)]
        x = x + pe_p
        return self.dropout(x)

#モデルに入力するために次元を拡張する
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Linear(c_in, d_model) 

    def forward(self, x):
        x = self.tokenConv(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers,
        d_model, d_input, d_output,
        dim_feedforward, dropout, nhead): #dim_ff = 512. dropout =0.1.
        
        super(Transformer, self).__init__()
        

        #エンべディングの定義
        self.token_embedding_src = TokenEmbedding(d_input, d_model)
        self.token_embedding_tgt = TokenEmbedding(d_output, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        #エンコーダの定義
        encoder_layer = TransformerEncoderLayer(d_model=d_model, 
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                                )
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 
                                                    num_layers=num_encoder_layers,
                                                    norm=encoder_norm
                                                    )
        
        #デコーダの定義
        decoder_layer = TransformerDecoderLayer(d_model=d_model, 
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                                )
        decoder_norm = LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(decoder_layer, 
                                                    num_layers=num_decoder_layers, 
                                                    norm=decoder_norm
                                                    )
        
        #出力層の定義
        #self.output = nn.Linear(d_model, d_output)
        
        self.fc = nn.Linear(d_model, d_output) #it was 18, 2.
        #self.fc1 = nn.Linear(d_model*d_model, 512)
        #self.fc2 = nn.Linear(512, 1)
        #self.sm = nn.Softmax(dim=1) #dim was 2. why??
        #self.sig = F.sigmoid(x)

    def forward(self, src, tgt, mask_src, mask_tgt):
        print("size of src in Transformer forwawrd.: {}".format(src.size()))
        token_embedding_src = self.token_embedding_src(src)
        embedding_src = self.positional_encoding(token_embedding_src)
        #print("embedding_src.")
        #print(embedding_src)
        #print(embedding_src.size())
        memory = self.transformer_encoder(embedding_src, mask_src)
        memory = np.squeeze(memory, 0)
        memory = memory.reshape(memory.shape[0], -1)
        #print("memory.")
        #print(memory)
        #print(memory.size())
        x = self.fc(memory)
        #x = self.fc1(memory)
        #x = self.fc2(x)
        #x = self.sm(x)
        x = F.sigmoid(x)
        return x


    def encode(self, src, mask_src):
        return self.transformer_encoder(self.positional_encoding(self.token_embedding_src(src)), mask_src)

    def decode(self, tgt, memory, mask_tgt):
        return self.transformer_decoder(self.positional_encoding(self.token_embedding_tgt(tgt)), memory, mask_tgt)
class NN(nn.Module):
    def __init__(self, d_model, d_input, d_output):
        self.d_model = d_model
        self.d_input = d_input
        self.d_output = d_output
        self.fc1 = nn.Linear(d_input, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc = nn.Linear(32, d_output)
    def forward(self, input_data):    
        x = self.fc1(input_data)
        x = self.fc2(x)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x


def create_mask(src, tgt, device):
    seq_len_src = src.shape[0]
    seq_len_tgt = tgt.shape[0]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt).to(device)
    mask_src = generate_square_subsequent_mask(seq_len_src).to(device)

    return mask_src, mask_tgt


def generate_square_subsequent_mask(seq_len):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask


def train(model, data_provider, optimizer, criterion, device, epoch):
    model.train()
    total_loss = []
    
    tgt_list_train = []
    output_list_train = []

    #correct_list_train = []
    i = 0
    for src, tgt in data_provider:

        print("epoch: {}, {}th train phase".format(epoch, h)+"="*20)
        
        src = src.float().to(device)
        tgt = tgt.float().to(device)

        #print("tgt")
        #print(tgt)

        input_tgt = None
        mask_src = None
        mask_tgt = None 
        #print("train src")
        #print(src)
        output = model(
            src=src, tgt=input_tgt, 
            mask_src=mask_src, mask_tgt=mask_tgt
        )
        #print("output in train.")
        #print(output)
        
        tgt_list_train.append(tgt)
        output_list_train.append(output)
        
        optimizer.zero_grad()


        loss = criterion(output, tgt)
        #print("loss in train.")
        #print(loss)
        #loss = criterion(output, tgt)
        #loss.retain_grad()
        
        loss.backward()

        total_loss.append(loss.cpu().detach())
        #print("total_loss in train.")
        #print(total_loss)

        optimizer.step()

        i += 1
        

        
    return np.average(total_loss), tgt_list_train, output_list_train

"""
def evaluate(flag, model, data_provider, criterion, device, epoch, l, protease):
    model.eval()
   total_loss = []

    tgt_list_valid = []
    output_list_valid = []

    #correct_list_valid = []

    with torch.no_grad():
        h = 0
        for src, tgt in data_provider:
            print("epochs: {}, {}th valid phase".format(epoch, h)+"="*20)
            
            src = src.float().to(device)
            tgt = tgt.float().to(device)

            seq_len_src = src.shape[1]

            #print("evaluate src")
            #print(src)
            #print(src.shape)

            tgt02 = torch.unsqueeze(tgt[-1], dim=0)

            mask_src = None
            mask_tgt = None
            output = model(src, tgt, mask_src, mask_tgt)
            #print("output in valid.")
            #print(output)
            output02 = output.tolist()
            #print("output02 in valid.")
            #print(output02)
            #print(len(output02))
            tgt_list_valid.append(tgt)
            output_list_valid.append(output)


            loss = criterion(output, tgt)
            total_loss.append(loss.cpu().detach())
            h = h + 1
        
    if flag=='test':
        true = torch.cat((src, tgt), dim=1)
        pred = torch.cat((src, output), dim=1)
        plt.plot(true.squeeze().cpu().detach().numpy(), label='true')
        plt.plot(pred.squeeze().cpu().detach().numpy(), label='pred')
        plt.legend()
        #plt.savefig('test.pdf')
        #"C:/Users/ysasaki.ADC/Documents/workstation_ysasaki/python/LSTM/VR-ML/MLprogram/labeled_data/flow2-2nd-move-data/outputdir"
        plt.savefig("./outputdir_rf_reg_priority/{}_{}/test.pdf".format(l, protease))
    #return np.average(total_loss), tgt_list_valid, output_list_valid, correct_list_valid
    return np.average(total_loss), tgt_list_valid, output_list_valid
"""

def predict(val_dataloader, model, device, model_path):
    test_dataloader = val_dataloader
    #model_path = "c:/Users/ysasaki.ADC/Documents/workstation_ysasaki/theme/NEDO-XRAI/MR3-ware/share_dataset_for_ysasaki/share_dataset_for_ysasaki/input_data7/outputdir/tf_checkpoint/3_2025-01-16_397756_transformer_checkpoint.pt"
    #model_path = "./outputdir/tf_checkpoint/"
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device))) 
    model.eval()
    with torch.no_grad():
        # output_all = torch.zeros(1, 4)
        output_all = np.zeros((1, 1))
        for i in range(len(test_dataloader)):
        #for j in range(1):
            #for i in range(len(test_dataloader)):
            #if j == 1:
            #    break
            if i == 0:
                batch_iterator = iter(test_dataloader)  # イテレータに変換, dataloaders_dict["test"]
            #print("="*20+"batch_iterator"+"="*20)
            #print(batch_iterator)
            tdata, labels = next(batch_iterator)  # 1番目の要素を取り出す
            #print("="*20+"test data {}".format(i)+"="*20)
            #print(tdata)
            #print(tdata.shape)
            print("="*20+"test labels {}: ".format(i)+"="*20)
            #print(labels)
            #print(len(labels))
            # GPUを使用する場合は明示的に指定
            src = tdata.to(device)
            tgt = labels.to(device)
            #tgt02 = torch.unsqueeze(tgt[-1], dim=0)
            #print("tgt02")
            #print(tgt02)
            #print("src[-1:]")
            #print(src[-1:])
            #input_tgt = torch.cat((src[-1:],tgt02), dim=1) # dim was 1.
            #print("input_tgt")
            #print(input_tgt)
            # 出力
            #mask_src, mask_tgt = create_mask(src, input_tgt, device)
            mask_src = None
            mask_tgt = None
            input_tgt = None 
            outputs = model(
                src=src, tgt=tgt, 
                mask_src=mask_src, mask_tgt=mask_tgt
            ) #順伝播
            outputs = np.array(outputs.to('cpu'))
            print(outputs)
            #outputs = torch.squeeze(outputs, dim=0)
            #output_all = torch.cat([output_all, outputs], axis=0)
            output_all = np.concatenate([output_all, outputs])
            #print("output_all in progress.")
            #print(output_all)
            #print(output_all.shape)
    output_all = output_all[1:]
    print("output_all.")
    print(output_all)
    print(output_all.shape)
    return output_all

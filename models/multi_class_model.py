import random
import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel

from sklearn.metrics import classification_report

class MultiClassModel(pl.LightningModule):
    def __init__(self, dropout, n_out, lr) -> None:
        super(MultiClassModel, self).__init__()
        
        # INISIALISASI
        torch.manual_seed(1)
        # Menentukan nilai random seed(1 angka) yang akan mempengaruhi matrik weight(bentuk matrik)
        random.seed(1)
        
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        # untuk transfer weight agar tidak hilang (di influence agar nilai berubah di tiap epoc)
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)
        # untuk melabeli total jumlah token ke dalam jumlah label yang dimiliki
        self.classifier = nn.Linear(768, n_out)
        
        self.lr = lr
        # menghitung loss function
        self.criterion = nn.BCEWithLogitsLoss
        
    # PROSES KALKULASI DAN INPUT
    def forward(self, input_ids, attention_mask, token_type_ids):
        #bert_ out = bert_output
        bert_out = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        # output layer terakhir
        hidden_state = bert_out[0]
        # mengambil output dengan output size() (batch size = 30 baris, sequence length = 100 kata/token, hidden_size = 768 tensor jumlah vektor representation)
        pooler = hidden_state[:, 0]
        # transfer weight ke epoch selanjutnya
        pooler = self.pre_classifier(pooler)
        
        
        # tanh 1 - (-1) // untuk mengkonversi nilai weight pada range 1 sampai -1
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        # classifier untuk memprojeksikan hasil pooler ke jumlah label
        output = self.classifier(pooler)
        
        return output
    
    def configure_optimizer(self):
        # adam ==> transfer weight, untuk nentuin weightnya tidak melenceng terlalu jauh(memberi sugestion), agar training lebih cepat, tidak memakan banyak memori, mengefisienkan random, kontrol loss
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch
        
        out = self(input_ids=x_input_ids, attention_mask=x_attention_mask, token_type_ids = x_token_type_ids)
        
        # ketiga parameter diinput dan dolah oleh method /function forward
        
        loss = self.criterion(out, target = y.float())
        
        pred = out.argmax(1).cpu()
        
        true = y.argmax(1).cpu()
        
        report = classification_report(true, pred, output_dict= true, zero_division = 0)
        
        self.log("accuracy", report["accuracy"], prog_bar = true)
        self.log("loss", loss)
        
        return loss
    
    def Validation_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch
        
        out = self(input_ids=x_input_ids, attention_mask=x_attention_mask, token_type_ids = x_token_type_ids)
        
        # ketiga parameter diinput dan dolah oleh method /function forward
        
        loss = self.criterion(out, target = y.float())
        
        pred = out.argmax(1).cpu()
        
        true = y.argmax(1).cpu()
        
        report = classification_report(true, pred, output_dict= true, zero_division = 0)
        
        self.log("accuracy", report["sccuracy"], prog_bar = true)
        self.log("loss", loss)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch
        
        out = self(input_ids=x_input_ids, attention_mask=x_attention_mask, token_type_ids = x_token_type_ids)
        
        # ketiga parameter diinput dan dolah oleh method /function forward
        
        
        pred = out.argmax(1).cpu()
        
        true = y.argmax(1).cpu()
        
        return pred, true
    
    # if __name__ == '__main__':
        # memastikan bahwa setiap random.seed menghasilkan weight yang sama
        # random.seed(1)
        # for i in range(10):
        #     print(random.random())
        
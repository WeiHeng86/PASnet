import torch
import torch.nn as nn


class PASNet(nn.Module):
    def __init__(self, In_Nodes, Gene_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, Gene_Mask, Pathway_Mask):
        super(PASNet, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
        self.gene_mask = Gene_Mask
        self.pathway_mask = Pathway_Mask
        ###SNP layer --> gene layer
        self.sc0 = nn.Linear(In_Nodes, Gene_Nodes)
        ###gene layer --> pathway layer
        self.sc1 = nn.Linear(Gene_Nodes, Pathway_Nodes)
        ###pathway layer --> hidden layer
        self.sc2 = nn.Linear(Pathway_Nodes, Hidden_Nodes)
        ###hidden layer --> Output layer
        self.sc3 = nn.Linear(Hidden_Nodes, Out_Nodes)
        ###randomly select a small sub-network
        self.do_m0 = torch.ones(Gene_Nodes)
        self.do_m1 = torch.ones(Pathway_Nodes)
        self.do_m2 = torch.ones(Hidden_Nodes)
        ###if gpu is being used
        if torch.cuda.is_available():
            self.do_m0 = self.do_m0.cuda()
            self.do_m1 = self.do_m1.cuda()
            self.do_m2 = self.do_m2.cuda()
        ###

    def forward(self, x):
        ###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
        self.sc0.weight.data = self.sc0.weight.data.mul(self.gene_mask)
        ###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
        self.sc1.weight.data = self.sc1.weight.data.mul(self.pathway_mask)
        x = self.sigmoid(self.sc0(x))
        if self.training == True:  ###construct a small sub-network for training only
            x = x.mul(self.do_m0)
        x = self.sigmoid(self.sc1(x))
        if self.training == True: ###construct a small sub-network for training only
            x = x.mul(self.do_m1)
        x = self.sigmoid(self.sc2(x))
        if self.training == True: ###construct a small sub-network for training only
            x = x.mul(self.do_m2)
        x = self.softmax(self.sc3(x)) # all rows add up to 1

        return x


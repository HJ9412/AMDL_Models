class Evaluator1(nn.Module):    # Unsupervised Evaluator
    def __init__(self, data_dim, hidden_dim1, hidden_dim2, value_dim):
        super(Evaluator, self).__init__()
        # Layers
        self.fc1 = nn.Linear(data_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.value = nn.Linear(hidden_dim2, value_dim)

    def forward(self, x):
        # Apply layers
        x = torch.flatten(x, start_dim = 1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        evaluation = F.relu(self.value(h2))
        return evaluation


#List of self-supervised evaluators
    
class Evaluator2(nn.Module):    # Reconstruction Feature Extractor
    def __init__(self, data_dim, hidden_dim1, hidden_dim2, value_dim):
        super(Evaluator, self).__init__()
        # Layers
        self.fc_x1h1 = nn.Linear(data_dim, hidden_dim1)
        self.fc_x2h2 = nn.Linear(data_dim, hidden_dim1)
        self.fc_h1A = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_h2A = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_h1B = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_h2B = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_eval = nn.Linear(hidden_dim2*2, value_dim)

    def forward(self, x1, x2):
        x1 = torch.flatten(x1, start_dim = 1)
        x2 = torch.flatten(x2, start_dim = 1)
        h1 = F.relu(self.fc_x1h1(x1))
        h2 = F.relu(self.fc_x2h2(x2))
        h1A = self.fc_h1A(h1)
        h2A = self.fc_h2A(h2)
        hA = F.relu(h1A + h2A)
        h1B = self.fc_h1B(h1)
        h2B = self.fc_h2B(h2)
        hB = F.relu(h1B + h2B)
        state = torch.concat((hA, hB), dim=1)
        values = F.relu(self.fc_eval(state))
        return values

class Evaluator3(nn.Module):    # Bias Eraser
    def __init__(self, data_dim, hidden_dim1, hidden_dim2, value_dim):
        super(Evaluator, self).__init__()
        # Layers
        self.fc_x1h1 = nn.Linear(data_dim, hidden_dim1)
        self.fc_x2h2 = nn.Linear(data_dim, hidden_dim1)
        self.fc_sA = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_sB = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_eval = nn.Linear(hidden_dim2, value_dim)

    def forward(self, x1, x2):
        x1 = torch.flatten(x1, start_dim = 1)
        x2 = torch.flatten(x2, start_dim = 1)
        h1 = F.relu(self.fc_x1h1(x1))               # Layers x1h1, x2h2 makes the evaluation a noncommutative operation
        h2 = F.relu(self.fc_x2h2(x2))
        sA = F.relu(self.fc_hA(h1)+self.fc_sA(h2))  # Applying common layers to h1 and h2 produces intermingled states sA and sB
        sB = F.relu(self.fc_sB(h1)+self.fc_sB(h2))
        values = F.relu(self.fc_eval(sA+sB))        # The last layer even intermingles the two states sA and sB to make it completely forget its pathway
        return values

class Evaluator4(nn.Module):    # Non-biased Feature Extractor
    def __init__(self, data_dim, hidden_dim1, hidden_dim2, value_dim):
        super(Evaluator, self).__init__()
        # Layers
        self.fc_x1h1 = nn.Linear(data_dim, hidden_dim1)
        self.fc_x2h2 = nn.Linear(data_dim, hidden_dim1)
        self.fc_h1A = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc_h2A = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc_h1B = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc_h2B = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc_state = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_eval = nn.Linear(hidden_dim2*2, value_dim)

    def forward(self, x1, x2):
        x1 = torch.flatten(x1, start_dim = 1)
        x2 = torch.flatten(x2, start_dim = 1)
        h1 = F.relu(self.fc_x1h1(x1))
        h2 = F.relu(self.fc_x2h2(x2))
        h1A = self.fc_h1A(h1)
        h2A = self.fc_h2A(h2)
        hA = F.relu(h1A + h2A)
        h1B = self.fc_h1B(h1)
        h2B = self.fc_h2B(h2)
        hB = F.relu(h1B + h2B)
        sA = F.relu(self.fc_state(hA))
        sB = F.relu(self.fc_state(hB))
        values = F.relu(self.fc_eval(torch.concat((sA,sB), dim=1)))
        return values

class Evaluator5(nn.Module):    # Non-biased Feature Extractor 2
    def __init__(self, data_dim, hidden_dim1, hidden_dim2, value_dim):
        super(Evaluator, self).__init__()
        # Layers
        self.fc_x0h0 = nn.Linear(data_dim, hidden_dim1)
        self.fc_x1h1 = nn.Linear(data_dim, hidden_dim1)
        self.fc_h0 = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc_h1 = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc_state = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_eval = nn.Linear(hidden_dim2*, value_dim)

    def forward(self, x0, x1):
        x0 = torch.flatten(x0, start_dim = 1)
        x1 = torch.flatten(x1, start_dim = 1)
        h0 = F.relu(self.fc_x0h0(x0))
        h1 = F.relu(self.fc_x1h1(x1))
        h00 = self.fc_h0(h0)
        h10 = self.fc_h0(h1)
        h01 = self.fc_h1(h0)
        h11 = self.fc_h1(h1)        # hij = fc_hj(hi). We want to erase i from it.
        hA = F.relu(h00 + h11)
        hB = F.relu(h10 + h01)
        # hC = F.relu(h00 - h11)    # {h00, h10, h01, h11} to {h00+h11, h10+h01, h00-h11, h10-h01}
        # hD = F.relu(h10 - h01)    # is a transformation of four vectors into non-factorable combinations.
                                    # It seems that adding hC and hD gives a freedom for the two 'fc_hj' to change its sign.
        sA = F.relu(self.fc_state(hA))
        sB = F.relu(self.fc_state(hB))
        # sC = F.relu(self.fc_state(hC))
        # sD = F.relu(self.fc_state(hD))
        values = F.relu(self.fc_eval(torch.concat((sA, sB), dim=1)))    # Any idea would get me helped a lot!
        return values


class Evaluator6(nn.Module):    # Non-biased Feature Extraction and then Evaluate
    def __init__(self, data_dim, hidden_dim1, hidden_dim2, value_dim):
        super(Evaluator, self).__init__()
        # Layers
        self.fc_x0h = nn.Linear(data_dim, hidden_dim1)
        self.fc_x1h = nn.Linear(data_dim, hidden_dim1)
        self.fc_hs0 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_hs1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_eval = nn.Linear(hidden_dim2, value_dim)

    def forward(self, x0, x1):
        x0 = torch.flatten(x0, start_dim = 1)
        x1 = torch.flatten(x1, start_dim = 1)
        h0 = F.relu(self.fc_x0h(x0))
        h1 = F.relu(self.fc_x1h(x1))
        s00 = self.fc_hs0(h0)
        s10 = self.fc_hs0(h1)
        s01 = self.fc_hs1(h0)
        s11 = self.fc_hs1(h1)
        sA = F.relu(s00 + s11)
        sB = F.relu(s01 + s10)
        vA = F.relu(self.fc_eval(sA))
        vB = F.relu(self.fc_eval(sB))
        values = vA + vB
        return values

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


#List of auto-supervised evaluators
    
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
        self.fc_eval = nn.Linear(hidden_dim2, value_dim)

    def forward(self, x1, x2):
        x1 = torch.flatten(x1, start_dim = 1)
        x2 = torch.flatten(x2, start_dim = 1)
        h1 = F.relu(self.fc_x1h1(x1))
        h2 = F.relu(self.fc_x2h2(x2))
        h1A = self.fc_h1A(h1)
        h2A = self.fc_h2A(h2)
        hA = h1A + h2A
        h1B = self.fc_h1B(h1)
        h2B = self.fc_h2B(h2)
        hB = h1B + h2B
        state = torch.concat((F.relu(hA), F.relu(hB)), dim=1)
        values = F.relu(self.fc_eval(state))
        return values

class Evaluator3(nn.Module):    # Bias Eraser
    def __init__(self, data_dim, hidden_dim1, hidden_dim2, value_dim):
        super(Evaluator, self).__init__()
        # Layers
        self.fc_x1h1 = nn.Linear(data_dim, hidden_dim1)
        self.fc_x2h2 = nn.Linear(data_dim, hidden_dim1)
        self.fc_hA = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_hB = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_eval = nn.Linear(hidden_dim2, value_dim)

    def forward(self, x1, x2):
        x1 = torch.flatten(x1, start_dim = 1)
        x2 = torch.flatten(x2, start_dim = 1)
        h1 = F.relu(self.fc_x1h1(x1))               # Layers x1h1, x2h2 makes the evaluation a noncommutative operation
        h2 = F.relu(self.fc_x2h2(x2))
        sA = F.relu(self.fc_hA(h1)+self.fc_hA(h2))  # Applying common layers to h1 and h2 produces intermingled states hA and hB
        sB = F.relu(self.fc_hB(h1)+self.fc_hB(h2))
        values = F.relu(self.fc_eval(sA+sB))        # The last layer even intermingles the two states hA and hB to make it completely forget its pathway
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

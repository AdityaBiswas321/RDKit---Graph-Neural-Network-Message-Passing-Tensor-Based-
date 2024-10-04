import rdkit
from rdkit import Chem
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear


# Access the PeriodicTable

electronegativities = {
    1: 2.20,   # Hydrogen
    6: 2.55,   # Carbon
    7: 3.04,   # Nitrogen
    8: 3.44,   # Oxygen
    9: 3.98,   # Fluorine
    # Add other elements as needed
}

def hybridization_to_int(hybridization):
    hybrid_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        # Handling less common types by mapping them to the closest common type
        Chem.rdchem.HybridizationType.SP3D: 2,
        Chem.rdchem.HybridizationType.SP3D2: 2
    }
    return hybrid_map.get(hybridization, 2)  # Default to SP3 if not found or unknown

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)  
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0, keepdim=True)
        x = self.lin(x)
        return x

# Initialize model, optimizer, and loss function
model = GCN(input_dim=3, hidden_dim=64, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Example dataset: List of tuples (SMILES string, label)
dataset = [('CCO', 1), ('CC', 0), ('OCCO', 0)]

# Training the model
for epoch in range(2000):
    total_loss = 0
    for smiles, label in dataset:
        molecule = Chem.MolFromSmiles(smiles)
        node_features = []
        for atom in molecule.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            electronegativity = electronegativities.get(atomic_num, None)
            hybrid_feat = hybridization_to_int(atom.GetHybridization())
            node_features.append([atomic_num, electronegativity, hybrid_feat])
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds()]
        edge_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()
        graph_data = Data(x=node_features, edge_index=edge_index)
        target = torch.tensor([label], dtype=torch.long)
        
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Average Loss: {total_loss / len(dataset)}")

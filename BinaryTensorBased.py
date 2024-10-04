import rdkit
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        Chem.rdchem.HybridizationType.SP3D: 2,
        Chem.rdchem.HybridizationType.SP3D2: 2
    }
    return hybrid_map.get(hybridization, 2)  # Default to SP3 if not found or unknown

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len + self.nbr_fea_len, 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        
        # Gather neighbor atom features
        atom_nbr_fea = atom_in_fea[nbr_fea_idx.view(-1)].view(N, M, -1)
        
        # Concatenate atom features, neighbor features, and bond features
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(-1, M, -1),
             atom_nbr_fea,
             nbr_fea], dim=2)
        
        # Fully connected layer followed by activation and normalization
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, 2*self.atom_fea_len)).view(N, M, 2*self.atom_fea_len)
        
        # Split into filter and core features
        nbr_filter, nbr_core = torch.chunk(total_gated_fea, 2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        
        # Summing the filtered core features
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        
        # Update atom features
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nbr_fea_len):
        super(GCN, self).__init__()
        # Embedding layer to project input features to hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.conv1 = ConvLayer(atom_fea_len=hidden_dim, nbr_fea_len=nbr_fea_len)
        self.conv2 = ConvLayer(atom_fea_len=hidden_dim, nbr_fea_len=nbr_fea_len)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx):
        # Embedding
        atom_fea = self.embedding(atom_fea)
        atom_fea = F.relu(atom_fea)
        
        # First convolution
        x = self.conv1(atom_fea, nbr_fea, nbr_fea_idx)
        x = F.relu(x)
        
        # Second convolution
        x = self.conv2(x, nbr_fea, nbr_fea_idx)
        x = F.relu(x)
        
        # Pooling: averaging over all atoms to get graph representation
        x = torch.mean(x, dim=0, keepdim=True)
        
        # Final output layer
        x = self.fc_out(x)
        return x

# Initialize model, optimizer, and loss function
model = GCN(input_dim=3, hidden_dim=64, output_dim=2, nbr_fea_len=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Example dataset: List of tuples (SMILES string, label)
dataset = [('CCO', 1), ('CC', 0), ('OCCO', 0)]

# Training the model
for epoch in range(200):
    total_loss = 0
    for smiles, label in dataset:
        molecule = Chem.MolFromSmiles(smiles)
        node_features = []
        bond_features = []
        bond_indices = []
        
        # Extract atom features
        for atom in molecule.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            electronegativity = electronegativities.get(atomic_num, 2.5)  # Default to 2.5 if unknown
            hybrid_feat = hybridization_to_int(atom.GetHybridization())
            node_features.append([atomic_num, electronegativity, hybrid_feat])
        
        # Convert node features to tensor
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Extract bond features (assuming a simple bond feature)
        for bond in molecule.GetBonds():
            bond_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bond_features.append([1])  # Placeholder for bond features
        
        # Handle cases where there are no bonds
        if bond_indices:
            bond_indices = torch.tensor(bond_indices, dtype=torch.long)
            bond_features = torch.tensor(bond_features, dtype=torch.float)
        else:
            # If no bonds, create empty tensors
            bond_indices = torch.empty((0, 2), dtype=torch.long)
            bond_features = torch.empty((0, 1), dtype=torch.float)
        
        N = len(node_features)
        M = max(len(bond_indices), 1)  # Ensure M is at least 1 to avoid zero-dimension tensors
        
        # Create neighbor feature indices and bond features per atom
        nbr_fea_idx = torch.zeros((N, M), dtype=torch.long)
        nbr_fea = torch.zeros((N, M, 1), dtype=torch.float)
        
        # Build neighbor indices and features
        for idx in range(N):
            neighbors = []
            bond_fea = []
            for bond_idx, (a, b) in enumerate(bond_indices.tolist()):
                if a == idx:
                    neighbors.append(b)
                    bond_fea.append(bond_features[bond_idx])
                elif b == idx:
                    neighbors.append(a)
                    bond_fea.append(bond_features[bond_idx])
            # Pad if necessary
            while len(neighbors) < M:
                neighbors.append(0)
                bond_fea.append(torch.zeros(1))
            nbr_fea_idx[idx] = torch.tensor(neighbors[:M])
            nbr_fea[idx] = torch.stack(bond_fea[:M])
        
        target = torch.tensor([label], dtype=torch.long)
        
        # Forward pass
        model.train()
        optimizer.zero_grad()
        out = model(node_features, nbr_fea, nbr_fea_idx)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Average Loss: {total_loss / len(dataset)}")

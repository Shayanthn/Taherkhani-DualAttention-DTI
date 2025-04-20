import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------
# Data Preparation
# ----------------------------

class DTIDataset:
    def __init__(self, data_path='article/data/davis/', max_protein_len=1200):
        """
        Load and preprocess Davis or KIBA dataset
        
        Args:
            data_path: Path to dataset directory
            max_protein_len: Maximum length of protein sequences (longer sequences will be truncated)
        """
        self.max_protein_len = max_protein_len
        
        # Load data (example for Davis dataset)
        self.df = pd.read_csv(os.path.join(data_path, 'data.csv'))
        self.df = self.df[['compound_iso_smiles', 'target_sequence', 'affinity']]
        self.df.columns = ['smiles', 'sequence', 'affinity']
        
        # Preprocess
        self.drug_graphs = []
        self.protein_seqs = []
        self.affinities = []
        
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Convert SMILES to graphs and protein sequences to indices"""
        valid_indices = []
        
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Preprocessing data"):
            smiles = row['smiles']
            seq = row['sequence']
            affinity = row['affinity']
            
            # Convert SMILES to graph
            drug_graph = self._smiles_to_graph(smiles)
            if drug_graph is None:
                continue
                
            # Convert protein sequence to indices
            protein_seq = self._protein_to_seq(seq)
            
            self.drug_graphs.append(drug_graph)
            self.protein_seqs.append(protein_seq)
            self.affinities.append(affinity)
            valid_indices.append(i)
        
        # Filter dataframe
        self.df = self.df.iloc[valid_indices]
        self.affinities = torch.tensor(self.affinities, dtype=torch.float)
        
    def _smiles_to_graph(self, smiles):
        """Convert SMILES string to molecular graph"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Atom features: [atomic_num, degree, formal_charge, hybridization, aromatic]
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetFormalCharge()),
                float(atom.GetHybridization().real),
                float(atom.GetIsAromatic())
            ]
            atom_features.append(features)
        
        # Edge features: [bond_type, conjugated, in_ring]
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_index.append((i, j))
            edge_index.append((j, i))  # Undirected graph
            
            bond_features = [
                float(bond.GetBondTypeAsDouble()),
                float(bond.GetIsConjugated()),
                float(bond.IsInRing())
            ]
            edge_attr.append(bond_features)
            edge_attr.append(bond_features)
        
        if len(edge_index) == 0:
            # Handle single-atom molecules
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _protein_to_seq(self, sequence):
        """Convert protein sequence to amino acid indices"""
        # Simple amino acid to index mapping
        aa_map = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
            'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }
        
        seq = [aa_map.get(aa, 20) for aa in sequence[:self.max_protein_len]]  # 20 for unknown
        if len(seq) < self.max_protein_len:
            seq += [21] * (self.max_protein_len - len(seq))  # 21 for padding
        
        return torch.tensor(seq, dtype=torch.long)
    
    def get_data_loaders(self, batch_size=32, test_size=0.2):
        """Create train and test data loaders"""
        train_idx, test_idx = train_test_split(
            range(len(self.drug_graphs)), 
            test_size=test_size, 
            random_state=42
        )
        
        # Create datasets
        train_data = [
            (self.drug_graphs[i], self.protein_seqs[i], self.affinities[i]) 
            for i in train_idx
        ]
        test_data = [
            (self.drug_graphs[i], self.protein_seqs[i], self.affinities[i]) 
            for i in test_idx
        ]
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader

# ----------------------------
# Model Architecture
# ----------------------------

class GNNProteinDTI(nn.Module):
    def __init__(self, drug_feat_dim=5, protein_embed_dim=64, hidden_dim=256):
        super().__init__()
        
        # Drug encoder (Graph Attention Network)
        self.drug_conv1 = GATConv(drug_feat_dim, 64, heads=2)
        self.drug_conv2 = GATConv(64*2, 128)
        self.drug_fc = nn.Linear(128, hidden_dim)
        
        # Protein encoder (CNN)
        self.protein_embed = nn.Embedding(22, protein_embed_dim)  # 20 AA + unknown + padding
        self.protein_conv1 = nn.Conv1d(protein_embed_dim, 64, kernel_size=3, padding=1)
        self.protein_conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.protein_fc = nn.Linear(128, hidden_dim)
        
        # Prediction head
        self.fc1 = nn.Linear(hidden_dim*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        
    def forward(self, drug_data, protein_seq):
        # Drug graph processing
        x, edge_index, batch = drug_data.x, drug_data.edge_index, drug_data.batch
        
        x = F.elu(self.drug_conv1(x, edge_index))
        x = F.elu(self.drug_conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        drug_feat = F.relu(self.drug_fc(x))
        
        # Protein sequence processing
        prot_embed = self.protein_embed(protein_seq)
        prot_embed = prot_embed.permute(0, 2, 1)  # (batch, channels, seq_len)
        
        prot_feat = F.elu(self.protein_conv1(prot_embed))
        prot_feat = F.elu(self.protein_conv2(prot_feat))
        prot_feat = F.adaptive_max_pool1d(prot_feat, 1).squeeze(2)
        prot_feat = F.relu(self.protein_fc(prot_feat))
        
        # Concatenate features
        combined = torch.cat([drug_feat, prot_feat], dim=1)
        
        # Prediction
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.out(x)

# ----------------------------
# Training and Evaluation
# ----------------------------

class Trainer:

    def plot_fc1_weights_heatmap(self):   
        import seaborn as sns
        import matplotlib.pyplot as plt

        fc_weights = self.model.fc1.weight.detach().cpu().numpy()  # shape: (128, 512)
        input_size = fc_weights.shape[1]
        mid = input_size // 2

        plt.figure(figsize=(14, 6))
        ax = sns.heatmap(fc_weights, cmap='viridis', cbar=True)

        plt.title("Heatmap of FC1 Weights (Feature Importance by Hidden Neurons)")
        plt.xlabel("Input Features")
        plt.ylabel("Hidden Layer Units")

        # تقسیم بندی Drug / Protein
        plt.axvline(x=mid, color='red', linestyle='--', linewidth=2)

        # برچسب‌های بخش‌های x-axis
        ax.text(mid / 2, -5, 'Drug Features', ha='center', va='center', fontsize=10, color='black')
        ax.text(mid + mid / 2, -5, 'Protein Features', ha='center', va='center', fontsize=10, color='black')

        plt.tight_layout()
        plt.show()

    def __init__(self, model, train_loader, test_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.test_losses = []
        self.rmses = []
        self.cis = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for drug_data, protein_seq, affinity in tqdm(self.train_loader, desc="Training"):
            drug_data = drug_data.to(self.device)
            protein_seq = protein_seq.to(self.device)
            affinity = affinity.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            pred = self.model(drug_data, protein_seq)
            loss = self.criterion(pred, affinity)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * drug_data.num_graphs
        
        return total_loss / len(self.train_loader.dataset)
    
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_pred = []
        all_true = []
        
        with torch.no_grad():
            for drug_data, protein_seq, affinity in tqdm(loader, desc="Evaluating"):
                drug_data = drug_data.to(self.device)
                protein_seq = protein_seq.to(self.device)
                affinity = affinity.to(self.device).unsqueeze(1)
                
                pred = self.model(drug_data, protein_seq)
                loss = self.criterion(pred, affinity)
                
                total_loss += loss.item() * drug_data.num_graphs
                all_pred.extend(pred.cpu().numpy().flatten())
                all_true.extend(affinity.cpu().numpy().flatten())
        
        rmse = np.sqrt(mean_squared_error(all_true, all_pred))
        ci = np.corrcoef(all_true, all_pred)[0, 1]
        return total_loss / len(loader.dataset), rmse, ci
    
    def train(self, epochs=50):
        for epoch in range(1, epochs+1):
            train_loss = self.train_epoch()
            test_loss, rmse, ci = self.evaluate(self.test_loader)
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.rmses.append(rmse)
            self.cis.append(ci)
            
            print(f'Epoch {epoch:02d}, Train Loss: {train_loss:.4f}, '
                  f'Test Loss: {test_loss:.4f}, RMSE: {rmse:.4f}, CI: {ci:.4f}')
    
    def plot_results(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title('Training and Test Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.rmses, label='RMSE')
        plt.plot(self.cis, label='CI')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.legend()
        plt.title('Evaluation Metrics')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_case_study(self, smiles, sequence, true_affinity):
        """Visualize a drug-protein interaction case"""
        self.model.eval()
        
        # Prepare inputs
        drug_graph = self._smiles_to_graph(smiles).to(self.device)
        protein_seq = self._protein_to_seq(sequence).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            pred_affinity = self.model(drug_graph, protein_seq).item()
        
        print(f"True affinity: {true_affinity:.2f}, Predicted: {pred_affinity:.2f}")
        
        # Visualize molecule
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol)
        
        return img, pred_affinity
    
    def _smiles_to_graph(self, smiles):
        """Helper function to convert SMILES to graph"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetFormalCharge()),
                float(atom.GetHybridization().real),
                float(atom.GetIsAromatic())
            ]
            atom_features.append(features)
        
        # Edge features
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append((i, j))
            edge_index.append((j, i))
        
        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        x = torch.tensor(atom_features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)
    
    def _protein_to_seq(self, sequence):
        """Helper function to convert protein sequence to indices"""
        aa_map = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
            'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }
        
        seq = [aa_map.get(aa, 20) for aa in sequence[:1200]]
        if len(seq) < 1200:
            seq += [21] * (1200 - len(seq))
        
        return torch.tensor(seq, dtype=torch.long)


# ----------------------------
# Main Execution
# ----------------------------

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    dataset = DTIDataset(data_path='article/data/davis/')
    train_loader, test_loader = dataset.get_data_loaders(batch_size=32)
    
    # Initialize model
    model = GNNProteinDTI()
    
    # Train and evaluate
    trainer = Trainer(model, train_loader, test_loader, device)
    print("Starting training...")
    trainer.train(epochs=50)
    print("Visualizing first layer weights as heatmap...")
    trainer.plot_fc1_weights_heatmap()

    
    # Plot results
    trainer.plot_results()
    
    # Case study example (using actual data from dataset if available)
    example_smiles = "CC1=C(C(=CC=C1)NC(=O)C2=CC(=NC=N2)NC3=NC(=CC=N3)C4=CC=CC=C4)Cl"
    example_seq = "MGCANCAGAFPRRGPPGARGGPGGP"  # Shortened for example
    example_affinity = 8.5
    
    print("\nRunning case study...")
    mol_img, pred_aff = trainer.visualize_case_study(
        example_smiles, example_seq, example_affinity
    )
    mol_img.show()
    
    # Save model
    torch.save(model.state_dict(), 'dti_model.pth')
    print("Model saved to dti_model.pth")

if __name__ == "__main__":
    main()
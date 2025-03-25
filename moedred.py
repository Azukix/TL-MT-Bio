import pandas as pd
from mordred import Calculator, descriptors
from mordred import Calculator, descriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem import PandasTools
from sklearn.preprocessing import StandardScaler 
# 读取txt文件为DataFrame
datatra = pd.read_csv('111.txt')



PandasTools.AddMoleculeColumnToFrame(datatra,'smiles','Molecule')
calc = Calculator(descriptors, ignore_3D=True)
X_train = pd.DataFrame(calc.pandas(datatra['Molecule']))
X_train.to_csv('mordred_results.csv', index=False)

# 将结果存储到csv文件

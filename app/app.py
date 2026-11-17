from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
import joblib

app = FastAPI()

sol_model = joblib.load("solubility_rf_model.pkl")
logp_model = joblib.load("logp_rf_model.pkl")
tpsa_model = joblib.load("tpsa_rf_model.pkl")

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

class MoleculeRequest(BaseModel):
    smiles: str

@app.post("/predict")
def predict_properties(request: MoleculeRequest):
    mol = Chem.MolFromSmiles(request.smiles)
    if not mol:
        raise HTTPException(status_code=400, detail="Invalid SMILES")
    
    fp = mfpgen.GetFingerprintAsNumPy(mol).reshape(1, -1)
    
    sol = float(sol_model.predict(fp)[0])
    logp = float(logp_model.predict(fp)[0])
    tpsa = float(tpsa_model.predict(fp)[0])
    
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_3d)
    sdf_block = Chem.MolToMolBlock(mol_3d)
    
    return {
        "solubility": sol,
        "logp": logp,
        "tpsa": tpsa,
        "sdf_block": sdf_block
    }
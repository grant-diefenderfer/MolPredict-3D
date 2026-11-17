import numpy as np, pandas as pd, joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

DESC_NAMES = [n for n, _ in Descriptors._descList]
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def featurize(smiles, exclude=()):
    names = [n for n in DESC_NAMES if n not in exclude]
    funcs = dict(Descriptors._descList)
    X, keep = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        desc = []
        for n in names:
            try:
                desc.append(funcs[n](mol))
            except Exception:
                desc.append(np.nan)
        fp = mfpgen.GetCountFingerprintAsNumPy(mol)
        X.append(np.concatenate([desc, fp]))
        keep.append(i)
    X = np.array(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    return X, np.array(keep)

def fit_and_report(X, y, name, save_as):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    candidates = {
        "RandomForest": RandomForestRegressor(n_estimators=300, max_features=0.33,
                                              min_samples_leaf=1, n_jobs=-1, random_state=42),
        "HistGB": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=500,
                                                l2_regularization=1.0, random_state=42),
    }
    cv = {k: cross_val_score(m, X_tr, y_tr, cv=5, scoring="r2").mean() for k, m in candidates.items()}
    best = max(cv, key=cv.get)
    model = candidates[best].fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = mean_squared_error(y_te, pred) ** 0.5
    print(f"{name}: best={best}, CV R2={cv[best]:.3f}, test R2={r2_score(y_te, pred):.3f}, RMSE={rmse:.3f}")
    joblib.dump(model, save_as, compress=3)

# Lipophilicity (logD) + TPSA
lipo = pd.read_csv("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv")
X, keep = featurize(lipo["smiles"], exclude=("TPSA",))
fit_and_report(X, lipo["exp"].values[keep], "Lipophilicity (logD)", "logd_model.pkl")

tpsa = np.array([Descriptors.TPSA(Chem.MolFromSmiles(s)) for s in lipo["smiles"].iloc[keep]])
fit_and_report(X, tpsa, "TPSA", "tpsa_model.pkl")

# ESOL solubility
esol = pd.read_csv("https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv")
X, keep = featurize(esol["smiles"])
fit_and_report(X, esol["measured log solubility in mols per litre"].values[keep], "ESOL", "solubility_model.pkl")
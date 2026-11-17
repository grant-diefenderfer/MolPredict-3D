import streamlit as st
import requests
import py3Dmol
from stmol import showmol

st.set_page_config(layout="wide", page_title="Molecular ML Viewer")

st.title("🧪 Molecular ML & 3D Viewer")

smiles_input = st.text_input("Enter SMILES string:", "CC(=O)OC1=CC=CC=C1C(=O)O")

if st.button("Analyze Molecule"):
    with st.spinner("Running models and generating 3D conformation..."):
        try:
            response = requests.post("http://api:8000/predict", json={"smiles": smiles_input})
            
            if response.status_code == 200:
                data = response.json()
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("📊 ML Predictions")
                    st.metric("Solubility (LogS)", f"{data['solubility']:.2f}")
                    st.metric("Lipophilicity (LogP)", f"{data['logp']:.2f}")
                    st.metric("Polar Surface Area (TPSA)", f"{data['tpsa']:.2f} Å²")
                    
                with col2:
                    st.subheader("🧬 3D Structure")
                    view = py3Dmol.view(width=700, height=450)
                    view.addModel(data['sdf_block'], 'sdf')
                    view.setStyle({'stick': {}})
                    view.setBackgroundColor('#0E1117')
                    view.zoomTo()
                    showmol(view, height=450, width=700)
            else:
                st.error("Invalid SMILES or API Error.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API. Ensure Docker is running.")
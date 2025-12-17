# app_final.py - PRODUCTION VERSION WITH YOUR ACTUAL MODEL
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import requests
import json

# Set page config
st.set_page_config(
    page_title="AI-Powered Rare Disease Diagnosis",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        color: #A23B72;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 6px solid;
    }
    .pathogenic {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left-color: #d32f2f;
    }
    .benign {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left-color: #388e3c;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2E86AB;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧬 AI-Powered Rare Disease Diagnosis</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #555;">Dissertation Project: Integrating AlphaFold Predictions with Machine Learning</p>', unsafe_allow_html=True)

# Load your actual data and model
@st.cache_resource
def load_resources():
    """Load your actual trained model and data"""
    resources = {}
    
    try:
        # Load your trained model
        resources['model'] = joblib.load('rare_disease_model.pkl')
        st.success("✅ Actual trained ML model loaded")
    except:
        st.warning("⚠️ Using demo model - upload your actual model file")
        from sklearn.ensemble import RandomForestClassifier
        resources['model'] = RandomForestClassifier(n_estimators=10, random_state=42)
    
    try:
        # Load your protein data
        with open('protein_data.pkl', 'rb') as f:
            resources['protein_data'] = pickle.load(f)
        st.success("✅ Your verified AlphaFold data loaded")
    except:
        # Fallback to your verified data
        resources['protein_data'] = {
            'GBA': {'disease': 'Gaucher Disease', 'avg_plddt': 93.25, 'plddt_std': 6.5, 
                   'low_conf_residues': 5, 'high_conf_residues': 510, 'total_residues': 536, 'confidence_ratio': 0.95},
            'CFTR': {'disease': 'Cystic Fibrosis', 'avg_plddt': 75.62, 'plddt_std': 19.8,
                    'low_conf_residues': 120, 'high_conf_residues': 980, 'total_residues': 1480, 'confidence_ratio': 0.66},
            'MECP2': {'disease': 'Rett Syndrome', 'avg_plddt': 56.59, 'plddt_std': 25.3,
                     'low_conf_residues': 150, 'high_conf_residues': 220, 'total_residues': 486, 'confidence_ratio': 0.45},
            'GAA': {'disease': 'Pompe Disease', 'avg_plddt': 91.88, 'plddt_std': 7.8,
                   'low_conf_residues': 8, 'high_conf_residues': 850, 'total_residues': 952, 'confidence_ratio': 0.89},
            'HEXA': {'disease': 'Tay-Sachs Disease', 'avg_plddt': 93.44, 'plddt_std': 6.2,
                    'low_conf_residues': 4, 'high_conf_residues': 500, 'total_residues': 529, 'confidence_ratio': 0.95}
        }
    
    try:
        # Load feature names
        with open('feature_names.pkl', 'rb') as f:
            resources['feature_names'] = pickle.load(f)
    except:
        resources['feature_names'] = ['af_avg_plddt', 'af_plddt_std', 'af_low_conf_residues', 
                                     'af_high_conf_residues', 'af_confidence_ratio',
                                     'variant_plddt', 'plddt_difference']
    
    return resources

# Load resources
resources = load_resources()
model = resources['model']
PROTEIN_DATA = resources['protein_data']
FEATURE_NAMES = resources['feature_names']

# Helper functions
def extract_position(mutation):
    """Extract amino acid position - same as in your training"""
    import re
    patterns = [
        r'p\.\w+?(\d+)\w+',
        r'(\d+)[A-Z]>[A-Z]',  
        r'[A-Z](\d+)[A-Z]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(mutation))
        if match:
            return int(match.group(1))
    return None

def calculate_variant_features(protein, mutation):
    """Calculate features EXACTLY as in your training pipeline"""
    position = extract_position(mutation)
    
    if position and position <= protein['total_residues']:
        # Same logic as in your calculate_variant_impact function
        if position <= 50 or position >= protein['total_residues'] - 50:
            variant_plddt = protein['avg_plddt'] * 0.9
        else:
            variant_plddt = protein['avg_plddt'] * 1.0
        
        # Add variation (same as training)
        variant_plddt += np.random.normal(0, 5)
        variant_plddt = max(20, min(95, variant_plddt))
    else:
        variant_plddt = protein['avg_plddt']
    
    plddt_difference = variant_plddt - protein['avg_plddt']
    
    # Return features in EXACT SAME ORDER as training
    return [
        protein['avg_plddt'],
        protein['plddt_std'],
        protein['low_conf_residues'],
        protein['high_conf_residues'],
        protein['confidence_ratio'],
        variant_plddt,
        plddt_difference
    ]

def get_alphafold_structure(uniprot_id):
    """Fetch actual AlphaFold data from API"""
    try:
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        return None
    return None

# Sidebar
st.sidebar.header("🔍 Input Variant")
st.sidebar.markdown("---")

protein_choice = st.sidebar.selectbox(
    "Select Protein:",
    list(PROTEIN_DATA.keys()),
    format_func=lambda x: f"{x} - {PROTEIN_DATA[x]['disease']}",
    help="Choose from the 5 rare disease proteins in your research"
)

mutation_input = st.sidebar.text_input(
    "Enter Mutation:",
    placeholder="e.g., L444P, p.Leu444Pro, 444A>G",
    help="Use standard genetic variant notation"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")
st.sidebar.write(f"**Algorithm:** {model.__class__.__name__}")
st.sidebar.write(f"**Features:** {len(FEATURE_NAMES)} structural metrics")
st.sidebar.write(f"**Proteins:** {len(PROTEIN_DATA)} rare diseases")

# Main content
if mutation_input and protein_choice:
    protein = PROTEIN_DATA[protein_choice]
    
    # Calculate features using YOUR training logic
    features = calculate_variant_features(protein, mutation_input)
    
    # Create DataFrame with correct feature names
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    
    # Make prediction with YOUR actual model
    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    
    pathogenic_prob = probabilities[1] * 100
    benign_prob = probabilities[0] * 100
    
    # Display results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Features", "🏗️ Structure", "📈 Insights"])
    
    with tab1:
        st.markdown("### 🤖 AI Prediction Result")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box pathogenic">
                <h2>🚨 Likely Pathogenic</h2>
                <p style="font-size: 1.3rem;">Confidence: <b>{pathogenic_prob:.1f}%</b></p>
                <p><i>This variant is predicted to disrupt protein function based on structural analysis.</i></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box benign">
                <h2>✅ Likely Benign</h2>
                <p style="font-size: 1.3rem;">Confidence: <b>{benign_prob:.1f}%</b></p>
                <p><i>This variant is predicted to have minimal impact on protein structure.</i></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence meters
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pathogenic Confidence", f"{pathogenic_prob:.1f}%")
            st.progress(int(pathogenic_prob/100))
        with col2:
            st.metric("Benign Confidence", f"{benign_prob:.1f}%")
            st.progress(int(benign_prob/100))
    
    with tab2:
        st.markdown("### 📊 Structural Feature Analysis")
        
        # Display all features
        for i, (feature_name, value) in enumerate(zip(FEATURE_NAMES, features)):
            st.markdown(f"""
            <div class="feature-card">
                <b>{feature_name.replace('af_', '').replace('_', ' ').title()}:</b> {value:.2f}
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 🎯 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(importance_df, use_container_width=True)
    
    with tab3:
        st.markdown("### 🏗️ AlphaFold 3D Structure")
        
        uniprot_ids = {'GBA': 'P04062', 'CFTR': 'P13569', 'MECP2': 'P51608', 
                      'GAA': 'P10253', 'HEXA': 'P06865'}
        
        uniprot_id = uniprot_ids.get(protein_choice)
        
        if uniprot_id:
            # Embed AlphaFold viewer
            st.components.v1.html(f"""
            <div style="width: 100%; height: 500px; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                <iframe 
                    src="https://alphafold.ebi.ac.uk/entry/{uniprot_id}" 
                    width="100%" 
                    height="500"
                    style="border: none;">
                </iframe>
            </div>
            """, height=500)
            
            st.markdown(f"[Open in AlphaFold Database](https://alphafold.ebi.ac.uk/entry/{uniprot_id})")
    
    with tab4:
        st.markdown("### 📈 Biological Insights")
        
        # Provide interpretation based on features
        st.write("**Interpretation:**")
        
        if features[6] < -10:  # plddt_difference
            st.write("- Significant decrease in structural confidence")
            st.write("- Mutation likely disrupts protein folding")
        elif features[5] < 70:  # variant_plddt
            st.write("- Low predicted structural confidence")
            st.write("- Potential functional impact")
        else:
            st.write("- Minimal structural impact predicted")
            st.write("- Variant likely tolerated")
        
        st.markdown("---")
        st.write("**Clinical Considerations:**")
        st.write("- This prediction is based on structural features only")
        st.write("- Always validate with clinical data and functional assays")
        st.write("- Consult clinical geneticists for medical decisions")

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to Your Dissertation App!
    
    This application combines:
    
    ### 🧬 **Your Research Components:**
    1. **AlphaFold Structural Data** - Verified pLDDT scores for 5 rare disease proteins
    2. **Machine Learning Model** - Your trained classifier (Random Forest/XGBoost)
    3. **Clinical Variant Analysis** - Pathogenicity predictions
    
    ### 🎯 **How to Use:**
    1. **Select** a protein from the sidebar
    2. **Enter** a mutation using standard notation
    3. **View** the AI prediction with confidence scores
    4. **Explore** structural features and 3D visualization
    
    ### 🔬 **Test Examples:**
    - **GBA** + **L444P** (Gaucher disease)
    - **CFTR** + **F508del** (Cystic fibrosis)
    - **MECP2** + **R255X** (Rett syndrome)
    """)
    
    # Show dataset summary
    st.markdown("### 📊 Your Research Dataset Summary")
    summary_data = {
        'Protein': [],
        'Disease': [],
        'Variants in Dataset': [],
        'Avg pLDDT': []
    }
    
    for gene, data in PROTEIN_DATA.items():
        summary_data['Protein'].append(gene)
        summary_data['Disease'].append(data['disease'])
        summary_data['Avg pLDDT'].append(f"{data['avg_plddt']:.1f}")
        # You can add actual variant counts if you have them
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px;'>
    <h3>🎓 Dissertation Project</h3>
    <p><b>AI-Powered Health App for Rare Disease Diagnosis Using AlphaFold Predictions</b></p>
    <p><small>This tool demonstrates the integration of structural biology with machine learning for clinical variant interpretation.</small></p>
</div>
""", unsafe_allow_html=True)

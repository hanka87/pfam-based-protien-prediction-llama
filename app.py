from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
from matplotlib import pyplot as plt

# Try to import TensorFlow with fallback options
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_IMPORTED = True
except ImportError:
    try:
        from keras.models import load_model
        from keras.preprocessing.sequence import pad_sequences
        TF_IMPORTED = True
    except ImportError:
        TF_IMPORTED = False

app = Flask(__name__)

# Configure upload folder for plots
UPLOAD_FOLDER = 'static/images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 1. Define the amino acid encoding dictionary (char_dict)
codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

char_dict = {code: i+1 for i, code in enumerate(codes)}
max_length = 100

# 2. Load your trained model if TensorFlow/Keras is available
model1 = None
if TF_IMPORTED:
    try:
        model1 = load_model('new_protien_model.h5')
    except Exception as e:
        print(f"Note: Could not load model. Running in analysis-only mode. Error: {str(e)}")
        TF_IMPORTED = False

# 3. Create and fit LabelEncoder
class_labels = [f'PF{str(i).zfill(5)}' for i in range(1, 101)]
le = LabelEncoder()
le.fit(class_labels)

def predict_and_analyze_protein(sequence, label_encoder):
    """Predict protein family and perform analysis"""
    if not TF_IMPORTED or model1 is None:
        return {
            'family_accession': 'MODEL_NOT_LOADED',
            'confidence_score': 0,
            'top5_predictions': [],
            'raw_predictions': np.zeros(len(class_labels))
        }
    
    # Preprocess the input sequence
    encoded_seq = [char_dict.get(code, 0) for code in sequence]
    padded_seq = pad_sequences([encoded_seq], maxlen=max_length, padding='post', truncating='post')
    
    # Make prediction
    pred = model1.predict(padded_seq)
    pred_class = np.argmax(pred, axis=1)
    family_accession = label_encoder.inverse_transform(pred_class)[0]
    confidence_score = np.max(pred) * 100  # Convert to percentage
    
    # Get top 5 predictions
    top5_indices = np.argsort(pred[0])[-5:][::-1]
    top5_families = label_encoder.inverse_transform(top5_indices)
    top5_scores = pred[0][top5_indices] * 100
    
    return {
        'family_accession': family_accession,
        'confidence_score': confidence_score,
        'top5_predictions': list(zip(top5_families, top5_scores)),
        'raw_predictions': pred[0]
    }

def analyze_protein_properties(sequence):
    """Calculate protein properties using BioPython"""
    try:
        analysis = ProteinAnalysis(sequence)
        return {
            'molecular_weight': analysis.molecular_weight(),
            'isoelectric_point': analysis.isoelectric_point(),
            'instability_index': analysis.instability_index(),
            'secondary_structure': analysis.secondary_structure_fraction(),
            'aromaticity': analysis.aromaticity(),
            'gravy': analysis.gravy(),
            'flexibility': analysis.flexibility()
        }
    except Exception as e:
        print(f"Error in protein analysis: {str(e)}")
        return None

def create_amino_acid_plot(sequence):
    """Generate amino acid composition plot"""
    analysis = ProteinAnalysis(sequence)
    aa_composition = analysis.get_amino_acids_percent()
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(*zip(*aa_composition.items()))
    plt.title("Amino Acid Composition", fontsize=14, pad=20)
    plt.xlabel("Amino Acid", fontsize=12)
    plt.ylabel("Percentage", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=10)
    
    # Save plot to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def create_secondary_structure_plot(secondary_structure):
    """Plot secondary structure fractions"""
    labels = ['Helix', 'Turn', 'Sheet']
    sizes = [secondary_structure[0]*100, secondary_structure[1]*100, secondary_structure[2]*100]
    
    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
            startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title("Secondary Structure Composition", fontsize=14, pad=20)
    
    # Save plot to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def create_property_radar(properties):
    """Create radar chart of protein properties"""
    # Select properties for radar chart
    radar_properties = {
        'Instability Index': properties['instability_index'] / 50,  # Scaled
        'Aromaticity': properties['aromaticity'] * 10,  # Scaled
        'GRAVY': (properties['gravy'] + 2) / 4,  # Scaled from -2 to 2
        'Molecular Weight': properties['molecular_weight'] / 100000,  # Scaled
        'Isoelectric Point': properties['isoelectric_point'] / 14  # Scaled
    }
    
    labels = list(radar_properties.keys())
    values = list(radar_properties.values())
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]  # Close the radar chart
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='skyblue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title('Protein Properties Radar Chart', fontsize=14, pad=20)
    
    # Save plot to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sequence = request.form.get('protein_sequence', '').strip().upper()
        
        if not sequence:
            return render_template('index.html', error="Please enter a protein sequence")
        
        # Validate sequence contains only valid amino acids
        valid_chars = set(codes)
        invalid_chars = [c for c in sequence if c not in valid_chars]
        if invalid_chars:
            return render_template('index.html', 
                                error=f"Invalid characters in sequence: {', '.join(set(invalid_chars))}")
        
        # Predict protein family
        prediction_result = predict_and_analyze_protein(sequence, le)
        
        # Analyze protein properties
        properties = analyze_protein_properties(sequence)
        if not properties:
            return render_template('index.html', 
                                error="Error analyzing protein properties")
        
        # Generate plots
        aa_plot = create_amino_acid_plot(sequence)
        ss_plot = create_secondary_structure_plot(properties['secondary_structure'])
        radar_plot = create_property_radar(properties)
        
        return render_template('index.html',
                            sequence=sequence,
                            prediction=prediction_result,
                            properties=properties,
                            aa_plot=aa_plot,
                            ss_plot=ss_plot,
                            radar_plot=radar_plot)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
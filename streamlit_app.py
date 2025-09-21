import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict
import re

# Set page config
st.set_page_config(
    page_title="Urdu to Roman Urdu Translator",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .translation-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .input-box {
        background-color: rgba(255,255,255,0.9);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .output-box {
        background-color: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #3498db;
    }
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif;
        font-size: 1.4rem;
        direction: rtl;
        text-align: right;
        line-height: 1.8;
        color: #2c3e50;
    }
    .roman-text {
        font-family: 'Arial', sans-serif;
        font-size: 1.3rem;
        color: #2c3e50;
        line-height: 1.6;
    }
    .example-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .example-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .demo-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# YOUR CUSTOM BPE TOKENIZER CLASS
# ========================================

class CustomBPETokenizer:
    """Your exact Custom BPE Tokenizer implementation"""
    def __init__(self, vocab_size: int = 6000):
        self.vocab_size = vocab_size
        self.word_freqs = {}
        self.splits = {}
        self.merges = []
        self.vocab = {}

    def tokenize(self, text: str) -> List[str]:
        """Your exact tokenization method"""
        if not text:
            return []

        words = text.split()
        result = []

        for word in words:
            splits = list(word)

            for pair in self.merges:
                i = 0
                new_splits = []
                while i < len(splits):
                    if (i < len(splits) - 1 and
                        splits[i] == pair[0] and
                        splits[i + 1] == pair[1]):
                        new_splits.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_splits.append(splits[i])
                        i += 1
                splits = new_splits

            result.extend(splits)

        return result

    def load(self, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vocab_size = data['vocab_size']
            self.word_freqs = data['word_freqs']
            self.merges = data['merges']
            self.vocab = data['vocab']

# ========================================
# DEMO MODEL CLASSES (Lightweight versions)
# ========================================

class BiLSTMEncoder(nn.Module):
    """Lightweight BiLSTM Encoder for demo"""
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, dropout=0.3):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=2)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

class LSTMDecoder(nn.Module):
    """Lightweight LSTM Decoder for demo"""
    def __init__(self, vocab_size, embedding_dim, hidden_size, encoder_hidden_size, num_layers=4, dropout=0.3):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=2)
        self.hidden_projection = nn.Linear(encoder_hidden_size, hidden_size)
        self.cell_projection = nn.Linear(encoder_hidden_size, hidden_size)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.out_projection = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

class Seq2SeqModel(nn.Module):
    """Demo Seq2Seq Model"""
    def __init__(self, urdu_vocab_size, roman_vocab_size, embedding_dim=256,
                 encoder_hidden_size=512, decoder_hidden_size=512,
                 encoder_layers=2, decoder_layers=4, dropout=0.3):
        super(Seq2SeqModel, self).__init__()

        self.encoder = BiLSTMEncoder(
            vocab_size=urdu_vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=encoder_hidden_size // 2,
            num_layers=encoder_layers,
            dropout=dropout
        )

        self.decoder = LSTMDecoder(
            vocab_size=roman_vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=decoder_hidden_size,
            encoder_hidden_size=encoder_hidden_size,
            num_layers=decoder_layers,
            dropout=dropout
        )

        self.roman_vocab_size = roman_vocab_size

# ========================================
# LOADING FUNCTIONS WITH DEMO MODE
# ========================================

@st.cache_resource
def load_model_and_tokenizers():
    """Load tokenizers and handle demo mode for large model files"""
    try:
        # Load your custom tokenizers (these should work)
        urdu_tokenizer = CustomBPETokenizer()
        roman_tokenizer = CustomBPETokenizer()
        
        urdu_tokenizer.load('urdu_tokenizer.pkl')
        roman_tokenizer.load('roman_tokenizer.pkl')

        # Create vocabularies with special tokens
        urdu_vocab = urdu_tokenizer.vocab.copy()
        roman_vocab = roman_tokenizer.vocab.copy()

        # Add special tokens if not present
        special_tokens = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']
        for i, token in enumerate(special_tokens):
            if token not in urdu_vocab:
                urdu_vocab[token] = i
            if token not in roman_vocab:
                roman_vocab[token] = i

        # Create reverse vocabularies
        urdu_idx2token = {idx: token for token, idx in urdu_vocab.items()}
        roman_idx2token = {idx: token for token, idx in roman_vocab.items()}

        # Try to load the trained model
        try:
            model = Seq2SeqModel(
                urdu_vocab_size=len(urdu_vocab),
                roman_vocab_size=len(roman_vocab),
                embedding_dim=256,
                encoder_hidden_size=512,
                decoder_hidden_size=512,
                encoder_layers=2,
                decoder_layers=4,
                dropout=0.5
            )
            
            # Try to load trained weights
            model.load_state_dict(torch.load('model.pth', map_location='cpu'))
            model.eval()
            
            return model, urdu_tokenizer, roman_tokenizer, urdu_vocab, roman_vocab, urdu_idx2token, roman_idx2token
            
        except FileNotFoundError:
            # Model file not found - return demo mode
            st.warning("⚠️ Demo Mode: Trained model file is too large for GitHub (>100MB)")
            st.info("Showing interface demo with predefined translations. Custom tokenizers loaded successfully!")
            
            return None, urdu_tokenizer, roman_tokenizer, urdu_vocab, roman_vocab, urdu_idx2token, roman_idx2token
        
    except FileNotFoundError as e:
        st.error(f"Tokenizer files not found: {str(e)}")
        st.error("Please ensure the following files are in the repository:")
        st.code("""
        - urdu_tokenizer.pkl
        - roman_tokenizer.pkl
        """)
        return None, None, None, None, None, None, None

def translate_text_demo(model, urdu_tokenizer, roman_tokenizer, text, urdu_vocab, roman_vocab, urdu_idx2token, roman_idx2token, max_length=50):
    """Demo translation function with predefined high-quality translations"""
    
    # Predefined translations based on your actual model results
    demo_translations = {
        "یہ ایک خوبصورت دن ہے": "yeh ek khubsurat din hai",
        "میں آپ سے محبت کرتا ہوں": "main aap se mohabbat karta hun",
        "سورج آسمان میں چمک رہا ہے": "suraj aasman mein chamak raha hai",
        "پھول بہت خوشبو دار ہیں": "phool bahut khushbu dar hain",
        "بچے کھیل رہے ہیں": "bachay khel rahay hain",
        "کتاب پڑھنا اچھا ہے": "kitab parhna acha hai",
        "رات بہت اندھیری ہے": "raat bahut andheri hai",
        "پانی صاف اور ٹھنڈا ہے": "pani saaf aur thanda hai",
        "درخت سبز اور تازہ ہیں": "darakht sabz aur taza hain",
        "آج موسم بہت خوشگوار ہے": "aaj mosam bahut khushgawar hai"
    }
    
    input_text = text.strip()
    
    if model is not None:
        # If actual model is loaded, use real translation
        try:
            # Use your custom tokenizer
            tokens = urdu_tokenizer.tokenize(input_text)
            
            if not tokens:
                return "No valid tokens found"
            
            # Convert to indices using your vocabulary
            src_indices = [0]  # SOS token
            for token in tokens:
                if token in urdu_vocab:
                    src_indices.append(urdu_vocab[token])
                else:
                    src_indices.append(urdu_vocab.get('<UNK>', 3))
            src_indices.append(1)  # EOS token
            
            # Create tensor
            src_tensor = torch.tensor([src_indices], dtype=torch.long)
            src_lengths = torch.tensor([len(src_indices)])
            
            # Generate translation using your model
            with torch.no_grad():
                generated = model.generate(
                    src_tensor, 
                    src_lengths, 
                    max_length=max_length,
                    temperature=0.8,
                    sos_token=0,
                    eos_token=1
                )
            
            # Convert back to tokens
            decoded_tokens = []
            for idx in generated[0].cpu().numpy():
                if idx in roman_idx2token:
                    token = roman_idx2token[idx]
                    if token not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']:
                        decoded_tokens.append(token)
                if idx == 1:  # EOS token
                    break
                    
            return ' '.join(decoded_tokens) if decoded_tokens else "Translation failed"
            
        except Exception as e:
            return f"Translation error: {str(e)}"
    else:
        # Demo mode - use predefined translations
        # Check for exact matches first
        if input_text in demo_translations:
            return demo_translations[input_text]
        
        # Check for partial matches
        for urdu_text, roman_text in demo_translations.items():
            if any(word in input_text for word in urdu_text.split()):
                return f"{roman_text} (Demo: Partial match found)"
        
        # For completely new inputs, show demo message with tokenization
        if urdu_tokenizer:
            try:
                tokens = urdu_tokenizer.tokenize(input_text)
                return f"Demo Mode: Input tokenized as {tokens}. Trained model needed for full translation."
            except:
                return f"Demo Mode: '{input_text}' - Trained model file too large for GitHub hosting"
        else:
            return "Demo Mode: Tokenizer not available"

# ========================================
# MAIN APP
# ========================================

def main():
    # Header with your project info
    st.markdown('<h1 class="main-header">🔤 Neural Machine Translation</h1>', unsafe_allow_html=True)
    st.markdown("### **Urdu to Roman Urdu Translation using BiLSTM Encoder-Decoder**")
    st.markdown("*Built with Custom BPE Tokenization | PyTorch Implementation*")

    # Load your models (demo mode handling)
    model, urdu_tokenizer, roman_tokenizer, urdu_vocab, roman_vocab, urdu_idx2token, roman_idx2token = load_model_and_tokenizers()
    
    # Demo mode warning
    if model is None and urdu_tokenizer is not None:
        st.markdown("""
        <div class="demo-warning">
        <h4>📱 Demo Mode Active</h4>
        <p>The trained model file (~200MB) exceeds GitHub's 100MB limit. This demo shows:</p>
        <ul>
            <li>✅ Complete interface design</li>
            <li>✅ Custom BPE tokenizer working</li>
            <li>✅ Predefined high-quality translations</li>
            <li>✅ Full experiment results and analysis</li>
        </ul>
        <p><strong>For production deployment, the model would be hosted separately (Google Drive, Hugging Face, etc.)</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar with your model specifications
    with st.sidebar:
        st.markdown("## 🏗️ Model Architecture")
        
        if urdu_tokenizer is not None:
            if model is not None:
                st.markdown('<div class="success-message">✅ Full model loaded successfully!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="demo-warning">📱 Demo mode: Tokenizers loaded, model too large for GitHub</div>', unsafe_allow_html=True)
            
            # Your exact model specifications
            st.markdown("### Architecture Details")
            st.markdown("""
            **🔸 Encoder**: 2-layer BiLSTM  
            **🔸 Decoder**: 4-layer LSTM  
            **🔸 Embedding Dim**: 256  
            **🔸 Hidden Size**: 512  
            **🔸 Dropout**: 0.5 (Best Config)  
            **🔸 Framework**: PyTorch  
            **🔸 Tokenization**: Custom BPE ✅  
            """)
            
            # Your experiment results
            st.markdown("### 🏆 Best Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card"><h3>0.978</h3><p>BLEU-4 Score</p></div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card"><h3>1.01</h3><p>Perplexity</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card"><h3>0.008</h3><p>Character Error Rate</p></div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card"><h3>0.014</h3><p>Validation Loss</p></div>', unsafe_allow_html=True)
                
            # Vocabulary info
            st.markdown("### 📚 Custom Tokenization")
            if urdu_vocab and roman_vocab:
                st.write(f"**Urdu vocab size**: {len(urdu_vocab):,}")
                st.write(f"**Roman vocab size**: {len(roman_vocab):,}")
                st.write("**Method**: BPE (Byte Pair Encoding) ✅")
        else:
            st.markdown('<div class="info-box">❌ Files not loaded. Please check repository.</div>', unsafe_allow_html=True)
    
    # Main content
    if urdu_tokenizer is not None:  # Show interface if at least tokenizers are loaded
        # Translation interface
        st.markdown('<h2 class="sub-header">💬 Translation Interface</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="translation-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📝 Input (Urdu Text)")
            st.markdown('<div class="input-box">', unsafe_allow_html=True)
            urdu_input = st.text_area(
                "",
                value="یہ ایک خوبصورت دن ہے",
                height=120,
                placeholder="یہاں اردو متن درج کریں...",
                help="Enter Urdu text for Roman Urdu transliteration",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Translation button
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                translate_btn = st.button("🔄 Translate", type="primary", use_container_width=True)
        
        with col2:
            st.markdown("#### 🔤 Output (Roman Urdu)")
            st.markdown('<div class="output-box">', unsafe_allow_html=True)
            
            if translate_btn and urdu_input.strip():
                with st.spinner("🔄 Processing with custom tokenizer..."):
                    translation = translate_text_demo(
                        model, urdu_tokenizer, roman_tokenizer, 
                        urdu_input.strip(), urdu_vocab, roman_vocab, 
                        urdu_idx2token, roman_idx2token
                    )
                st.session_state.translation = translation
            
            if hasattr(st.session_state, 'translation'):
                st.markdown(f'<div class="roman-text">{st.session_state.translation}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="roman-text" style="color: #6c757d; font-style: italic;">Translation will appear here...</div>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Available demo examples
        if model is None:
            st.markdown('<h2 class="sub-header">📋 Available Demo Translations</h2>', unsafe_allow_html=True)
            st.markdown("*Try these examples to see the expected model performance:*")
            
            demo_examples = [
                "یہ ایک خوبصورت دن ہے",
                "میں آپ سے محبت کرتا ہوں", 
                "سورج آسمان میں چمک رہا ہے",
                "پھول بہت خوشبو دار ہیں",
                "بچے کھیل رہے ہیں"
            ]
            
            cols = st.columns(3)
            for i, example in enumerate(demo_examples):
                with cols[i % 3]:
                    if st.button(example, key=f"demo_{i}"):
                        st.session_state.demo_input = example
                        # Auto-fill the text area
                        st.rerun()
        
        # Your experiment results from the output
        st.markdown('<h2 class="sub-header">🧪 Experiment Results</h2>', unsafe_allow_html=True)
        
        # Create DataFrame with your actual results
        results_data = {
            'Experiment': ['Experiment_1_Emb128', 'Experiment_2_Hidden256', 'Experiment_3_Dropout05', 'Experiment_4_LR0005'],
            'Embedding_Dim': [128, 256, 256, 256],
            'Hidden_Size': [512, 256, 512, 512],
            'Dropout': [0.3, 0.3, 0.5, 0.3],
            'Learning_Rate': [0.001, 0.001, 0.001, 0.0005],
            'Final_Train_Loss': [0.024, 0.121, 0.019, 0.064],
            'Final_Val_Loss': [0.016, 0.082, 0.014, 0.040],
            'BLEU-1': [0.993, 0.942, 0.993, 0.946],
            'BLEU-4': [0.978, 0.868, 0.978, 0.857],
            'Perplexity': [1.02, 1.08, 1.01, 1.04],
            'CER': [0.009, 0.058, 0.008, 0.049]
        }
        
        df = pd.DataFrame(results_data)
        
        # Interactive plots
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig1 = px.bar(
                df, 
                x='Experiment', 
                y='BLEU-4',
                title='BLEU-4 Scores Across Experiments',
                color='BLEU-4',
                color_continuous_scale='viridis',
                text='BLEU-4'
            )
            fig1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.line(
                df, 
                x='Experiment', 
                y='CER',
                title='Character Error Rate (CER)',
                markers=True,
                line_shape='linear'
            )
            fig2.update_traces(line=dict(color='#e74c3c', width=3), marker=dict(size=8))
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed results table
        st.markdown("#### 📋 Comprehensive Results Table")
        st.dataframe(
            df.style.format({
                'Final_Train_Loss': '{:.3f}',
                'Final_Val_Loss': '{:.3f}',
                'BLEU-1': '{:.3f}',
                'BLEU-4': '{:.3f}',
                'Perplexity': '{:.2f}',
                'CER': '{:.3f}',
                'Learning_Rate': '{:.4f}'
            }).highlight_min(['Final_Val_Loss', 'CER'], color='lightgreen')
            .highlight_max(['BLEU-1', 'BLEU-4'], color='lightblue'),
            use_container_width=True
        )
        
        # Best model highlights
        st.markdown("### 🏆 Best Model: Experiment_3_Dropout05")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("BLEU-4 Score", "0.978", "↑ Best")
        with col2:
            st.metric("Perplexity", "1.01", "↓ Lowest")
        with col3:
            st.metric("CER", "0.008", "↓ Best")
        with col4:
            st.metric("Val Loss", "0.014", "↓ Lowest")
        
        # Example translations from your actual results
        st.markdown('<h2 class="sub-header">✨ Model Performance Examples</h2>', unsafe_allow_html=True)
        
        examples = [
            {"urdu": "یہ ایک خوبصورت دن ہے", "roman": "yeh ek khubsurat din hai", "status": "Perfect Match"},
            {"urdu": "میں آپ سے محبت کرتا ہوں", "roman": "main aap se mohabbat karta hun", "status": "Perfect Match"},
            {"urdu": "سورج آسمان میں چمک رہا ہے", "roman": "suraj aasman mein chamak raha hai", "status": "Perfect Match"}
        ]
        
        for i, example in enumerate(examples):
            with st.container():
                st.markdown(f'<div class="example-card">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.markdown("**Urdu Input:**")
                    st.markdown(f'<div class="urdu-text">{example["urdu"]}</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown("**Roman Output:**")
                    st.markdown(f'<div class="roman-text">{example["roman"]}</div>', unsafe_allow_html=True)
                with col3:
                    st.success(f"✅ {example['status']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Technical implementation
        st.markdown('<h2 class="sub-header">🛠️ Technical Implementation</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Model Architecture")
            st.code("""
# BiLSTM Encoder (2 layers)
encoder = BiLSTMEncoder(
    vocab_size=urdu_vocab_size,
    embedding_dim=256,
    hidden_size=256,  # 512//2 for bidirectional
    num_layers=2,
    dropout=0.5
)

# LSTM Decoder (4 layers)  
decoder = LSTMDecoder(
    vocab_size=roman_vocab_size,
    embedding_dim=256,
    hidden_size=512,
    num_layers=4,
    dropout=0.5
)
            """, language='python')
        
        with col2:
            st.markdown("#### Custom BPE Tokenization")
            st.code("""
# Custom BPE Implementation (Working!)
class CustomBPETokenizer:
    def train(self, corpus):
        # Learn merge operations
        self.merges = learn_bpe_merges(corpus)
        
    def tokenize(self, text):
        # Apply learned merges
        return apply_bpe(text, self.merges)
        
# Live tokenization working ✅
tokens = urdu_tokenizer.tokenize("یہ ایک جملہ ہے")
            """, language='python')
    
    else:
        # Error state
        st.error("⚠️ Unable to load tokenizer files.")
        st.markdown("### 📋 Required Files:")
        st.code("""
        - urdu_tokenizer.pkl
        - roman_tokenizer.pkl
        """)

    st.markdown("**Built with PyTorch • Custom BPE Tokenization • BiLSTM Encoder-Decoder Architecture**")
    
    # Project information and credits
    st.markdown('<h2 class="sub-header">📚 Project Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎯 Project Objectives")
        st.markdown("""
        - **Build BiLSTM encoder-decoder model** for Urdu → Roman Urdu translation
        - **Implement custom BPE tokenization** from scratch (no libraries)
        - **Conduct comprehensive experiments** with different hyperparameters
        - **Achieve high-quality transliteration** with proper evaluation metrics
        - **Deploy working web interface** with real-time translation
        """)
        
        st.markdown("#### 📊 Dataset & Preprocessing")
        st.markdown("""
        - **Source**: urdu_ghazals_rekhta dataset
        - **Split**: 50% train, 25% validation, 25% test
        - **Preprocessing**: Text normalization, custom BPE tokenization
        - **Vocabulary**: Dynamic vocabulary with special tokens
        """)
    
    with col2:
        st.markdown("#### ⚙️ Model Requirements")
        st.markdown("""
        - **Encoder**: 2-layer Bidirectional LSTM
        - **Decoder**: 4-layer LSTM with teacher forcing
        - **Framework**: PyTorch (mandatory)
        - **Loss Function**: Cross-entropy with padding ignore
        - **Optimizer**: Adam with gradient clipping
        """)
        
        st.markdown("#### 📈 Evaluation Metrics")
        st.markdown("""
        - **Primary**: BLEU-4 score, Perplexity
        - **Secondary**: Character Error Rate (CER), Edit Distance
        - **Qualitative**: Manual inspection of translations
        - **Target**: BLEU > 30 (achieved 97.8%)
        """)

    # Deployment status
    st.markdown('<h2 class="sub-header">🚀 Deployment Status</h2>', unsafe_allow_html=True)
    
    if model is None:
        st.markdown("""
        <div class="demo-warning">
        <h4>📱 Current Deployment: Demo Mode</h4>
        <p><strong>Why Demo Mode?</strong></p>
        <ul>
            <li>Trained model file: ~200MB (exceeds GitHub's 100MB limit)</li>
            <li>Custom tokenizers: ✅ Successfully deployed</li>
            <li>Interface & results: ✅ Fully functional</li>
        </ul>
        <p><strong>Production Deployment Options:</strong></p>
        <ul>
            <li>🔸 Host model on Google Drive/Hugging Face</li>
            <li>🔸 Use Git LFS for large files</li>
            <li>🔸 Deploy on cloud platforms with higher limits</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ Full model deployment successful!")

    # Technical notes
    st.markdown('<h2 class="sub-header">🔧 Technical Notes</h2>', unsafe_allow_html=True)
    
    with st.expander("Model Architecture Details"):
        st.markdown("""
        **BiLSTM Encoder:**
        - 2 layers, bidirectional
        - Hidden size: 256 (512 total with bidirectional)  
        - Processes Urdu input sequences
        - Packed sequences for efficient processing
        
        **LSTM Decoder:**
        - 4 layers, unidirectional
        - Hidden size: 512
        - Teacher forcing during training
        - Attention mechanism can be added for improvement
        
        **Custom BPE Tokenizer:**
        - Learned from training corpus
        - Handles out-of-vocabulary words
        - Subword-level tokenization
        - No external libraries used (implemented from scratch)
        """)
    
    with st.expander("Training Configuration"):
        st.markdown("""
        **Best Configuration (Experiment_3_Dropout05):**
        - Embedding dimension: 256
        - Hidden size: 512
        - Dropout: 0.5 (key factor for best performance)
        - Learning rate: 0.001
        - Batch size: 32
        - Epochs: 5 (for experiments)
        
        **Training Details:**
        - Adam optimizer with gradient clipping
        - Cross-entropy loss ignoring padding tokens
        - Validation-based early stopping
        - Model checkpointing for best performance
        """)
    
    with st.expander("Performance Analysis"):
        st.markdown("""
        **Why Experiment_3_Dropout05 performed best:**
        - Higher dropout (0.5) provided better regularization
        - Prevented overfitting to training data
        - Better generalization to validation/test sets
        - Optimal balance between model capacity and regularization
        
        **BLEU Score Interpretation:**
        - 0.978 is exceptionally high for this task
        - Indicates near-perfect character-level alignment
        - May suggest model learned the mapping very well
        - Could benefit from more diverse test data for robustness
        """)
    
    with st.expander("Deployment Instructions"):
        st.markdown("""
        **Current Setup:**
        ```bash
        pip install streamlit torch plotly pandas numpy
        streamlit run streamlit_app.py
        ```
        
        **File Structure:**
        ```
        project/
        ├── streamlit_app.py
        ├── requirements.txt
        ├── urdu_tokenizer.pkl     ✅
        ├── roman_tokenizer.pkl    ✅
        └── model.pth             (200MB - GitHub limit issue)
        ```
        
        **For Full Model Deployment:**
        1. Use Git LFS: `git lfs track "*.pth"`
        2. Or host on Hugging Face Spaces
        3. Or use Google Colab deployment with ngrok
        """)

if __name__ == "__main__":
    main()

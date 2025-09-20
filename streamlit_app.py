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
# YOUR MODEL ARCHITECTURE
# ========================================

class BiLSTMEncoder(nn.Module):
    """Your exact BiLSTM Encoder implementation"""
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

    def forward(self, input_seq, lengths):
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)

        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

        return output, (hidden, cell)

class LSTMDecoder(nn.Module):
    """Your exact LSTM Decoder implementation"""
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

    def forward(self, input_token, hidden, cell):
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.out_projection(output)
        return output, hidden, cell

    def init_hidden(self, encoder_hidden, encoder_cell):
        batch_size = encoder_hidden.size(1)

        if encoder_hidden.size(0) < self.num_layers:
            encoder_hidden = encoder_hidden[-1:].repeat(self.num_layers, 1, 1)
            encoder_cell = encoder_cell[-1:].repeat(self.num_layers, 1, 1)
        elif encoder_hidden.size(0) > self.num_layers:
            encoder_hidden = encoder_hidden[-self.num_layers:]
            encoder_cell = encoder_cell[-self.num_layers:]

        hidden = self.hidden_projection(encoder_hidden)
        cell = self.cell_projection(encoder_cell)
        return hidden, cell

class Seq2SeqModel(nn.Module):
    """Your exact Seq2Seq Model implementation"""
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

    def generate(self, urdu_seq, urdu_lengths, max_length=100, temperature=1.0, sos_token=0, eos_token=1):
        """Your exact generation method"""
        self.eval()
        with torch.no_grad():
            batch_size = urdu_seq.size(0)
            encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(urdu_seq, urdu_lengths)
            decoder_hidden, decoder_cell = self.decoder.init_hidden(encoder_hidden, encoder_cell)

            decoder_input = torch.zeros(batch_size, 1, dtype=torch.long).to(urdu_seq.device)
            generated_sequences = []

            for t in range(max_length):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(
                    decoder_input, decoder_hidden, decoder_cell
                )

                probs = F.softmax(decoder_output.squeeze(1) / temperature, dim=-1)
                decoder_input = torch.multinomial(probs, 1)
                generated_sequences.append(decoder_input)

                if (decoder_input == eos_token).all():
                    break

            if generated_sequences:
                return torch.cat(generated_sequences, dim=1)
            else:
                return torch.zeros(batch_size, 1, dtype=torch.long).to(urdu_seq.device)

# ========================================
# LOADING FUNCTIONS
# ========================================


@st.cache_resource
def load_model_and_tokenizers():
    """Load your trained model and custom tokenizers"""
    try:
        # Load your custom tokenizers - UPDATED PATHS
        urdu_tokenizer = CustomBPETokenizer()
        roman_tokenizer = CustomBPETokenizer()
        
        # Look for files in root directory (since you uploaded directly)
        urdu_tokenizer.load('urdu_tokenizer.pkl')  # Removed processed_data/
        roman_tokenizer.load('roman_tokenizer.pkl')  # Removed processed_data/

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

        # Load your best model
        model = Seq2SeqModel(
            urdu_vocab_size=len(urdu_vocab),
            roman_vocab_size=len(roman_vocab),
            embedding_dim=256,
            encoder_hidden_size=512,
            decoder_hidden_size=512,
            encoder_layers=2,
            decoder_layers=4,
            dropout=0.3
        )
        
        # Load trained weights - look in root directory
        model_paths = [
            'best_model.pth'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
                model_loaded = True
                st.success(f"Model loaded successfully: {model_path}")
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                st.warning(f"Failed to load {model_path}: {str(e)}")
                continue
        
        if not model_loaded:
            st.error("Could not load any model file")
            return None, None, None, None, None, None, None
        
        return model, urdu_tokenizer, roman_tokenizer, urdu_vocab, roman_vocab, urdu_idx2token, roman_idx2token
        
    except FileNotFoundError as e:
        st.error(f"Tokenizer files not found: {str(e)}")
        st.error("Please ensure these files are in the root directory:")
        st.code("""
        - urdu_tokenizer.pkl
        - roman_tokenizer.pkl  
        - best_model.pth
        """)
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None, None, None
        
        return model, urdu_tokenizer, roman_tokenizer, urdu_vocab, roman_vocab, urdu_idx2token, roman_idx2token
        
    except FileNotFoundError as e:
        st.error(f"Model files not found: {str(e)}")
        st.error("Please ensure the following files are in the correct directory:")
        st.code("""
        - /urdu_tokenizer.pkl
        - /roman_tokenizer.pkl  
        - best_model.pth
        """)
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None, None, None
        

def translate_text(model, urdu_tokenizer, roman_tokenizer, text, urdu_vocab, roman_vocab, urdu_idx2token, roman_idx2token, max_length=50):
    """Translate Urdu text using your exact pipeline"""
    if model is None:
        return "Model not loaded"
    
    try:
        # Use your custom tokenizer
        tokens = urdu_tokenizer.tokenize(text.strip())
        
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
        st.error(f"Translation error: {str(e)}")
        return "Translation error occurred"

# ========================================
# MAIN APP
# ========================================

def main():
    # Header with your project info
    st.markdown('<h1 class="main-header">🔤 Neural Machine Translation</h1>', unsafe_allow_html=True)
    st.markdown("### **Urdu to Roman Urdu Translation using BiLSTM Encoder-Decoder**")
    st.markdown("*Built with Custom BPE Tokenization | PyTorch Implementation*")

    # Load your models
    model, urdu_tokenizer, roman_tokenizer, urdu_vocab, roman_vocab, urdu_idx2token, roman_idx2token = load_model_and_tokenizers()
    
    # Sidebar with your model specifications
    with st.sidebar:
        st.markdown("## 🏗️ Model Architecture")
        
        if model is not None:
            st.markdown('<div class="success-message">✅ Model loaded successfully!</div>', unsafe_allow_html=True)
            
            # Your exact model specifications
            st.markdown("### Architecture Details")
            st.markdown("""
            **🔸 Encoder**: 2-layer BiLSTM  
            **🔸 Decoder**: 4-layer LSTM  
            **🔸 Embedding Dim**: 256  
            **🔸 Hidden Size**: 512  
            **🔸 Dropout**: 0.5 (Best Config)  
            **🔸 Framework**: PyTorch  
            **🔸 Tokenization**: Custom BPE  
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
                st.write("**Method**: BPE (Byte Pair Encoding)")
        else:
            st.markdown('<div class="info-box">❌ Model not loaded. Please check file paths.</div>', unsafe_allow_html=True)
    
    # Main content
    if model is not None:
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
                with st.spinner("🔄 Translating with BiLSTM model..."):
                    translation = translate_text(
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
        
        # Perplexity comparison
        fig3 = px.bar(
            df,
            x='Experiment',
            y='Perplexity',
            title='Perplexity Comparison',
            color='Perplexity',
            color_continuous_scale='plasma',
            text='Perplexity'
        )
        fig3.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig3.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
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
        
        # Example translations from your output
        st.markdown('<h2 class="sub-header">✨ Model Examples</h2>', unsafe_allow_html=True)
        
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
        
        # Technical details
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
# Custom BPE Implementation
class CustomBPETokenizer:
    def train(self, corpus):
        # Learn merge operations
        self.merges = learn_bpe_merges(corpus)
        
    def tokenize(self, text):
        # Apply learned merges
        return apply_bpe(text, self.merges)
        
# Usage
tokens = urdu_tokenizer.tokenize("یہ ایک جملہ ہے")
            """, language='python')
        
    else:
        # Error state
        st.error("⚠️ Unable to load the translation model.")
        st.markdown("### 📋 Required Files:")
        st.code("""
├── urdu_tokenizer.pkl      # Your custom BPE tokenizer for Urdu
├── roman_tokenizer.pkl     # Your custom BPE tokenizer for Roman Urdu
└── ...

Experiment_3_Dropout05_model.pth  # Your best trained model weights
        """)
        
        st.markdown("### 🔧 Setup Instructions:")
        st.markdown("""
        1. **Upload your trained model files** to the correct directory
        2. **Ensure tokenizer files** are in the `processed_data/` folder
        3. **Refresh the app** to load the model
        """)

    # Footer
    st.markdown("---")
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

    # Deployment information
    st.markdown('<h2 class="sub-header">🚀 Deployment Guide</h2>', unsafe_allow_html=True)
    
    st.markdown("#### Running this Streamlit App")
    
    tab1, tab2, tab3 = st.tabs(["Local Setup", "File Requirements", "Usage Instructions"])
    
    with tab1:
        st.markdown("**1. Install Dependencies:**")
        st.code("""
pip install streamlit torch plotly pandas numpy
        """, language='bash')
        
        st.markdown("**2. Save this app as `app.py`**")
        
        st.markdown("**3. Run the application:**")
        st.code("""
streamlit run app.py
        """, language='bash')
        
        st.markdown("**4. Access at:** `http://localhost:8501`")
    
    with tab2:
        st.markdown("**Required files structure:**")
        st.code("""
project/
├── app.py                              # This Streamlit app
├── processed_data/
│   ├── urdu_tokenizer.pkl             # Your custom BPE tokenizer
│   ├── roman_tokenizer.pkl            # Your custom BPE tokenizer
│   └── dataset_stats.json             # Dataset statistics
├── Experiment_3_Dropout05_model.pth   # Best model weights
└── experiment_results.csv              # Experiment comparison data
        """)
        
        st.info("Make sure to save your tokenizers and model weights after training!")
    
    with tab3:
        st.markdown("**Using the Translation Interface:**")
        st.markdown("""
        1. **Enter Urdu text** in the input box (left side)
        2. **Click 'Translate'** to get Roman Urdu output
        3. **View results** in the output box (right side)
        4. **Explore experiment results** in the analysis section
        5. **Check model examples** for quality assessment
        """)
        
        st.markdown("**Supported Input:**")
        st.markdown("- Modern Urdu text with proper Unicode encoding")
        st.markdown("- Poetry and prose (trained on ghazals dataset)")
        st.markdown("- Short to medium length sentences work best")

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
        - No external libraries used
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
        - Could benefit from more diverse test data
        """)

if __name__ == "__main__":
    main()

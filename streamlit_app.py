@st.cache_resource
def load_model_and_tokenizers():
    """Load your trained model and custom tokenizers"""
    try:
        # Load your custom tokenizers
        urdu_tokenizer = CustomBPETokenizer()
        roman_tokenizer = CustomBPETokenizer()
        
        # FIXED PATHS - removed the incorrect leading slash
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

        # Load your best model
        model = Seq2SeqModel(
            urdu_vocab_size=len(urdu_vocab),
            roman_vocab_size=len(roman_vocab),
            embedding_dim=256,
            encoder_hidden_size=512,
            decoder_hidden_size=512,
            encoder_layers=2,
            decoder_layers=4,
            dropout=0.3  # Using 0.3 for Experiment_4_LR0005 which was your best
        )
        
        # Load trained weights - corrected model paths
        model_paths = [
            'best_model.pth',
            'Experiment_4_LR0005_model.pth',
            'Experiment_3_Dropout05_model.pth'
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
                st.success(f"✅ Model loaded successfully: {model_path}")
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
        st.error(f"Model files not found: {str(e)}")
        st.error("Please ensure the following files are in the root directory:")
        st.code("""
        - urdu_tokenizer.pkl
        - roman_tokenizer.pkl  
        - best_model.pth
        """)
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None, None, None

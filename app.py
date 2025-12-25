#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
import threading
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
import queue
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app)

# ============================================
# Configuration
# ============================================

MODEL_DIR = './addition_model'
MAX_NUMBER = 10000
MAX_SEQ_LENGTH = 250

# Construction dynamique du vocabulaire
def build_vocab():
    vocab = {}
    idx = 0
    for digit in '0123456789':
        vocab[digit] = idx
        idx += 1
    for sign in ['+', '=', '-']:
        vocab[sign] = idx
        idx += 1
    for special in ['<START>', '<END>', '<PAD>']:
        vocab[special] = idx
        idx += 1
    vocab[' '] = idx
    idx += 1
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        vocab[letter] = idx
        idx += 1
    return vocab

VOCAB = build_vocab()
VOCAB_SIZE = len(VOCAB)
REVERSE_VOCAB = {v: k for k, v in VOCAB.items()}

# Variables globales
encoder = None
decoder = None
training_history = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

terminal_logs = queue.Queue(maxsize=1000)

def log_to_terminal(message, level='info'):
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': str(message)
    }
    try:
        terminal_logs.put_nowait(log_entry)
    except queue.Full:
        try:
            terminal_logs.get_nowait()
            terminal_logs.put_nowait(log_entry)
        except:
            pass

def set_num_threads(num_threads: int):
    if num_threads is None:
        return
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)

def get_available_cores() -> int:
    return os.cpu_count() or 1

def tokenize(text: str) -> list:
    return [VOCAB.get(char, VOCAB['<PAD>']) for char in text]

def detokenize(tokens: list) -> str:
    return ''.join(
        REVERSE_VOCAB.get(t, '')
        for t in tokens
        if REVERSE_VOCAB.get(t) not in ['<START>', '<END>', '<PAD>', None]
    )

def pad_sequence(seq: list, max_len: int, pad_value: int = None) -> list:
    if pad_value is None:
        pad_value = VOCAB['<PAD>']
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))

def decompose_addition(a: int, b: int) -> str:
    result = a + b
    
    max_digits = max(len(str(a)), len(str(b)))
    str_a = str(a).zfill(max_digits)
    str_b = str(b).zfill(max_digits)
    
    partial_results = []
    for i in range(max_digits):
        power = max_digits - 1 - i
        multiplier = 10 ** power
        digit_a = int(str_a[i])
        digit_b = int(str_b[i])
        val_a = digit_a * multiplier
        val_b = digit_b * multiplier
        partial_sum = val_a + val_b
        partial_results.append((val_a, val_b, partial_sum))
    
    steps = []
    for val_a, val_b, partial_sum in partial_results:
        steps.append(f"on a {val_a}+{val_b}={partial_sum}")
    
    if len(partial_results) == 1:
        return f"{steps[0]} donc le resultat est {result}"
    
    running_total = partial_results[0][2]
    combined_steps = [steps[0]]
    
    for i in range(1, len(partial_results)):
        current_partial = partial_results[i][2]
        new_total = running_total + current_partial
        combined_steps.append(f"on a {running_total}+{current_partial}={new_total}")
        running_total = new_total
    
    all_steps = steps + combined_steps[1:]
    return " et ".join(all_steps) + f" donc le resultat est {result}"

def generate_training_data(num_samples: int) -> tuple:
    inputs, outputs = [], []
    for _ in range(num_samples):
        digits_a, digits_b = random.randint(1, 4), random.randint(1, 4)
        a, b = random.randint(0, (10**digits_a) - 1), random.randint(0, (10**digits_b) - 1)
        inputs.append(f"{a}+{b}=")
        outputs.append(decompose_addition(a, b))
    simple_samples = num_samples // 5
    for _ in range(simple_samples):
        a, b = random.randint(0, 99), random.randint(0, 99)
        inputs.append(f"{a}+{b}=")
        outputs.append(decompose_addition(a, b))
    for _ in range(simple_samples):
        a, b = random.randint(100, 999), random.randint(100, 999)
        inputs.append(f"{a}+{b}=")
        outputs.append(decompose_addition(a, b))
    combined = list(zip(inputs, outputs))
    random.shuffle(combined)
    inputs, outputs = zip(*combined) if combined else ([], [])
    return list(inputs), list(outputs)

class AdditionDataset(Dataset):
    def __init__(self, inputs: list, outputs: list):
        self.encoder_inputs = [pad_sequence(tokenize(i), MAX_SEQ_LENGTH) for i in inputs]
        self.decoder_inputs = [pad_sequence([VOCAB['<START>']] + tokenize(o), MAX_SEQ_LENGTH) for o in outputs]
        self.decoder_outputs = [pad_sequence(tokenize(o) + [VOCAB['<END>']], MAX_SEQ_LENGTH) for o in outputs]
    def __len__(self): return len(self.encoder_inputs)
    def __getitem__(self, idx):
        return (torch.tensor(self.encoder_inputs[idx], dtype=torch.long),
                torch.tensor(self.decoder_inputs[idx], dtype=torch.long),
                torch.tensor(self.decoder_outputs[idx], dtype=torch.long))

class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=VOCAB['<PAD>'])
        self.pos_encoding = nn.Parameter(torch.randn(1, MAX_SEQ_LENGTH, embed_dim))
        self.dropout = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    def forward(self, x):
        embedded = self.dropout(self.embedding(x) + self.pos_encoding[:, :x.size(1), :])
        return self.transformer_encoder(embedded)

class Decoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=VOCAB['<PAD>'])
        self.pos_encoding = nn.Parameter(torch.randn(1, MAX_SEQ_LENGTH, embed_dim))
        self.dropout = nn.Dropout(0.1)
        layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_linear = nn.Linear(embed_dim, vocab_size)
    def forward(self, x, encoder_outputs):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        embedded = self.dropout(self.embedding(x) + self.pos_encoding[:, :x.size(1), :])
        output = self.transformer_decoder(embedded, encoder_outputs, tgt_mask=tgt_mask)
        return self.output_linear(output)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
    def forward(self, encoder_input, decoder_input):
        return self.decoder(decoder_input, self.encoder(encoder_input))

def train(num_epochs=50, batch_size=32, num_samples=5000, num_threads=None):
    global encoder, decoder, training_history
    if num_threads: set_num_threads(num_threads)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    model = Seq2Seq(encoder, decoder).to(device)
    inputs, outputs = generate_training_data(num_samples)
    split = int(len(inputs) * 0.9)
    train_loader = DataLoader(AdditionDataset(inputs[:split], outputs[:split]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AdditionDataset(inputs[split:], outputs[split:]), batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB['<PAD>'])
    best_val_loss = float('inf')
    training_history = []
    os.makedirs(MODEL_DIR, exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for enc_in, dec_in, dec_out in train_loader:
            enc_in, dec_in, dec_out = enc_in.to(device), dec_in.to(device), dec_out.to(device)
            optimizer.zero_grad()
            output = model(enc_in, dec_in).view(-1, VOCAB_SIZE)
            loss = criterion(output, dec_out.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            mask = dec_out.view(-1) != VOCAB['<PAD>']
            correct += (output.argmax(dim=1)[mask] == dec_out.view(-1)[mask]).sum().item()
            total += mask.sum().item()
        avg_val_loss = 0
        model.eval()
        with torch.no_grad():
            for enc_in, dec_in, dec_out in val_loader:
                enc_in, dec_in, dec_out = enc_in.to(device), dec_in.to(device), dec_out.to(device)
                output = model(enc_in, dec_in).view(-1, VOCAB_SIZE)
                avg_val_loss += criterion(output, dec_out.view(-1)).item()
        avg_val_loss /= len(val_loader) if val_loader else 1
        training_history.append({'epoch': epoch+1, 'loss': total_loss/len(train_loader), 'accuracy': correct/total, 'val_loss': avg_val_loss})
        
        # Affichage terminal HTML (simulé avec log_to_terminal)
        progress = int((epoch + 1) / num_epochs * 30)
        progress_bar = '█' * progress + '░' * (30 - progress)
        epoch_log = f"  {epoch + 1:5d}  │  {total_loss/len(train_loader):.4f}  │   {correct/total * 100:5.2f}%    │   {avg_val_loss:.4f}   │  [{progress_bar}] {int((epoch + 1) / num_epochs * 100)}%"
        log_to_terminal(epoch_log, 'success')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, 'encoder.pt'))
            torch.save(decoder.state_dict(), os.path.join(MODEL_DIR, 'decoder.pt'))
    return model

def predict(input_str: str, temperature: float = 0.7) -> dict:
    global encoder, decoder
    if not encoder or not decoder: return {}
    encoder.eval(); decoder.eval()
    if not input_str.endswith('='): input_str += '='
    tokens = pad_sequence(tokenize(input_str), MAX_SEQ_LENGTH)
    enc_in = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        enc_out = encoder(enc_in)
        out_tokens, steps = [], []
        generated_text = ""
        for i in range(MAX_SEQ_LENGTH):
            dec_in = torch.tensor([pad_sequence([VOCAB['<START>']] + out_tokens, MAX_SEQ_LENGTH)], dtype=torch.long).to(device)
            logits = decoder(dec_in, enc_out)[0, len(out_tokens)]
            prob = torch.softmax(logits / temperature, dim=0)
            idx = prob.argmax().item()
            if idx == VOCAB['<END>'] or idx == VOCAB['<PAD>']:
                break
            token_char = REVERSE_VOCAB.get(idx, '?')
            out_tokens.append(idx)
            generated_text += token_char
            steps.append({'step': i+1, 'token': token_char, 'confidence': f"{prob[idx].item()*100:.2f}"})
            if "donc le resultat est" in generated_text:
                remaining_digits = 0
                for j in range(i + 1, min(i + 20, MAX_SEQ_LENGTH)):
                    dec_in = torch.tensor([pad_sequence([VOCAB['<START>']] + out_tokens, MAX_SEQ_LENGTH)], dtype=torch.long).to(device)
                    logits = decoder(dec_in, enc_out)[0, len(out_tokens)]
                    prob = torch.softmax(logits / temperature, dim=0)
                    idx = prob.argmax().item()
                    if idx == VOCAB['<END>'] or idx == VOCAB['<PAD>']:
                        break
                    token_char = REVERSE_VOCAB.get(idx, '?')
                    if token_char.isdigit():
                        remaining_digits += 1
                        out_tokens.append(idx)
                        generated_text += token_char
                        steps.append({'step': j+1, 'token': token_char, 'confidence': f"{prob[idx].item()*100:.2f}"})
                    elif remaining_digits > 0:
                        break
                    else:
                        out_tokens.append(idx)
                        generated_text += token_char
                        steps.append({'step': j+1, 'token': token_char, 'confidence': f"{prob[idx].item()*100:.2f}"})
                break
    return {'output': input_str + detokenize(out_tokens), 'steps': steps}

def save_model():
    if not encoder or not decoder: return False
    torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, 'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join(MODEL_DIR, 'decoder.pt'))
    with open(os.path.join(MODEL_DIR, 'history.json'), 'w') as f: json.dump(training_history, f)
    return True

def load_model():
    global encoder, decoder, training_history
    try:
        encoder, decoder = Encoder().to(device), Decoder().to(device)
        encoder.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'encoder.pt'), map_location=device))
        decoder.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'decoder.pt'), map_location=device))
        return True
    except: return False

def extract_final_result(decomp):
    if "donc le resultat est " in decomp:
        res = decomp.split("donc le resultat est ")[-1].strip()
        return "".join(c for c in res if c.isdigit())
    return ""

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({'model_loaded': encoder is not None, 'device': str(device), 'vocab_size': VOCAB_SIZE, 'max_seq_length': MAX_SEQ_LENGTH, 'training_epochs': len(training_history)})

@app.route('/api/train', methods=['POST'])
def api_train():
    data = request.json
    threading.Thread(target=train, kwargs={'num_epochs': data.get('epochs', 50), 'num_samples': data.get('samples', 5000)}).start()
    return jsonify({'status': 'started'})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    expr = request.json.get('expression', '')
    res = predict(expr)
    if not res: return jsonify({'error': 'no model'}), 500
    decomp = res['output'].split('=')[-1]
    expected = sum(int(x) for x in expr.replace('=', '').split('+') if x.isdigit())
    final = extract_final_result(decomp)
    return jsonify({'expression': expr, 'decomposition': decomp, 'final_result': final, 'expected': expected, 'correct': final == str(expected), 'steps': res['steps']})

@app.route('/api/test', methods=['POST'])
def api_test():
    n = request.json.get('num_tests', 10)
    results, correct = [], 0
    for _ in range(n):
        a, b = random.randint(0, 9999), random.randint(0, 9999)
        expr = f"{a}+{b}"
        res = predict(expr)
        if res:
            decomp = res['output'].split('=')[-1]
            final = extract_final_result(decomp)
            is_correct = final == str(a+b)
            if is_correct: correct += 1
            results.append({'expression': expr, 'expected': a+b, 'predicted': final, 'correct': is_correct})
    return jsonify({'results': results, 'correct': correct, 'total': n, 'accuracy': correct/n*100})

@app.route('/api/save', methods=['POST'])
def api_save(): return jsonify({'success': save_model()})

@app.route('/api/load', methods=['POST'])
def api_load(): return jsonify({'success': load_model()})

@app.route('/api/terminal-stream')
def stream():
    def gen():
        while True:
            try: yield f"data: {json.dumps(terminal_logs.get(timeout=1))}\n\n"
            except: yield f"data: {json.dumps({'ping': True})}\n\n"
    return app.response_class(gen(), mimetype='text/event-stream')

@app.route('/api/delete', methods=['POST'])
def api_delete():
    import shutil
    try:
        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)
        global encoder, decoder, training_history
        encoder, decoder = None, None
        training_history = []
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download')
def api_download():
    import zipfile
    import io
    if not os.path.exists(MODEL_DIR):
        return jsonify({'error': 'Aucun modèle à télécharger'}), 400
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(MODEL_DIR):
            for file in files:
                zf.write(os.path.join(root, file), file)
    
    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='model.zip')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

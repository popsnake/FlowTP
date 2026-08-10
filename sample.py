import os
import torch
import configparser
import numpy as np
import random
import math
from collections import Counter, defaultdict
from model.flowz import FlowMatcher
from utils.utils import set_seed
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from torch import device
from torch.cuda import is_available
from Bio import SeqIO, Align
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from statistics import mean, stdev

sample_type = 'afp' 
assert sample_type.lower() in ['amp', 'afp', 'avp']

cfs = 1.5
n = 1
sample_num = 1000
device = device("cuda:0" if is_available() else "cpu")
CFS_SCHEDULE = 'warmup_decay'
conf = configparser.ConfigParser()
conf.read('config.ini')
conf_dict = {k.lower(): v for k, v in conf.items('FlowTP_conf')}
set_seed(int(conf_dict.get('seed', 2024)))

temp_conf = {k.lower(): v for k, v in conf.items('temperature_cfs')}

sample_type = sample_type.lower()

t_min = float(temp_conf.get(f'{sample_type}_temperature_min', 0.9))
t_max = float(temp_conf.get(f'{sample_type}_temperature_max', 1.1))
CFS_MIN = float(temp_conf.get(f'{sample_type}_cfs_min', 1.5))
CFS_MAX = float(temp_conf.get(f'{sample_type}_cfs_max', 3.0))

USE_DYNAMIC_CFS = True
labels = {'antimicrobial': 0, 'antifungal': 1, 'antiviral': 2}
peptide_file = ''

class DynamicCFSScheduler:
    def __init__(self, schedule_type='linear', cfs_min=0.5, cfs_max=2.0):
        self.schedule_type = schedule_type
        self.cfs_min = cfs_min
        self.cfs_max = cfs_max

    def get_cfs(self, step, num_steps):
        progress = float(step) / float(max(1, num_steps - 1))

        if self.schedule_type == 'linear':
            cfs = self.cfs_min + (self.cfs_max - self.cfs_min) * progress
        elif self.schedule_type == 'cosine':
            cfs = self.cfs_min + (self.cfs_max - self.cfs_min) * (
                0.5 - 0.5 * math.cos(math.pi * progress)
            )
        elif self.schedule_type == 'quadratic':
            cfs = self.cfs_min + (self.cfs_max - self.cfs_min) * (progress ** 2)
        elif self.schedule_type == 'sigmoid':
            sigmoid_val = 1.0 / (1.0 + math.exp(-10 * (progress - 0.5)))
            cfs = self.cfs_min + (self.cfs_max - self.cfs_min) * sigmoid_val
        elif self.schedule_type == 'warmup_decay':
            if progress < 0.5:
                cfs = self.cfs_min + (self.cfs_max - self.cfs_min) * (progress * 2)
            else:
                decay_progress = (progress - 0.5) * 2
                cfs = self.cfs_max - (self.cfs_max - self.cfs_min) * decay_progress
        elif self.schedule_type == 'constant':
            cfs = self.cfs_max
        else:
            cfs = self.cfs_max

        return float(max(self.cfs_min, min(self.cfs_max, cfs)))

class LengthSampler:
    def __init__(self, path, max_len=254):
        def load_fasta_file(file_path):
            sequences = []
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Fasta file not found: {file_path}")
            with open(file_path, "r") as fasta_file:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    sequences.append(str(record.seq))
            return sequences

        data = load_fasta_file(path)
        lens = np.clip([len(t) for t in data], a_min=0, a_max=max_len)
        freqs = Counter(lens)
        self.distrib = np.array([freqs.get(i, 0) for i in range(max_len + 1)], dtype=float)
        
        total = self.distrib.sum()
        if total <= 0:
            min_len = max(1, int(np.median(lens)) - 10) if len(lens) > 0 else 1
            max_len_fallback = max(3, int(np.median(lens)) + 10) if len(lens) > 0 else min(max_len, 50)
            rng = np.zeros(max_len + 1, dtype=float)
            rng[min_len:max_len_fallback+1] = 1.0
            self.distrib = rng / rng.sum()
        else:
            self.distrib = self.distrib / total

    def sample(self, num_samples):
        draws = np.random.multinomial(1, self.distrib, size=num_samples)
        s = np.argmax(draws, axis=1)
        return s

def flow_matching_sample(flow_matcher, x_shape, num_steps=200, len_list=None, use_attention=True,
                         cfs=1., peptide_type='antimicrobial', return_attn_list=False, index=1,
                         class_label=None, activity_label=None, t_min=1, t_max=1.1,
                         temp_schedule='linear', clamp_value=20.0, use_midpoint=True,
                         dynamic_cfs=False, cfs_schedule_type='linear', cfs_min=0.5, cfs_max=2.0):
    
    batch_size = int(x_shape[0])
    unconditional_class = torch.zeros(batch_size, device=device, dtype=torch.int) + 3
    unconditional_activity = torch.zeros(batch_size, device=device, dtype=torch.int)

    conditional_class = torch.zeros(batch_size, device=device, dtype=torch.int) + (class_label if class_label is not None else 0)
    conditional_activity = torch.zeros(batch_size, device=device, dtype=torch.int) + (activity_label if activity_label is not None else 0)

    attn_list = None

    if len_list is None:
        length_sampler = LengthSampler(
            path=f'data/train/{peptide_file}.fasta',
            max_len=int(conf_dict['max_length'])
        )
        ones_counts = length_sampler.sample(batch_size) + 2
    else:
        ones_counts = np.array(len_list) + 2

    if len(ones_counts) != batch_size:
        batch_size = len(ones_counts)
        unconditional_class = torch.zeros(batch_size, device=device, dtype=torch.int) + 3
        unconditional_activity = torch.zeros(batch_size, device=device, dtype=torch.int)
        conditional_class = torch.zeros(batch_size, device=device, dtype=torch.int) + (class_label if class_label is not None else 0)
        conditional_activity = torch.zeros(batch_size, device=device, dtype=torch.int) + (activity_label if activity_label is not None else 0)

    attention_mask = np.zeros((len(ones_counts), int(conf_dict['max_length']) + 2), dtype=int)
    for i, count in enumerate(ones_counts):
        attention_mask[i, :count] = 1
    attention_mask = torch.from_numpy(attention_mask).to(device)

    try:
        x_shape[0] = len(ones_counts)
    except Exception:
        pass

    if dynamic_cfs:
        cfs_scheduler = DynamicCFSScheduler(
            schedule_type=cfs_schedule_type,
            cfs_min=cfs_min,
            cfs_max=cfs_max
        )
    else:
        cfs_scheduler = None

    x_t = torch.randn(x_shape).to(device)
    dt = 1.0 / max(1, int(num_steps))

    for step in tqdm(range(num_steps), desc=f'index {index} flow matching sampling', total=num_steps):
        progress = float(step) / float(max(1, num_steps - 1))

        if temp_schedule == 'linear':
            temperature = t_min + (t_max - t_min) * progress
        elif temp_schedule == 'cosine':
            temperature = t_min + (t_max - t_min) * (0.5 - 0.5 * math.cos(math.pi * progress))
        elif temp_schedule == 'quadratic':
            temperature = t_min + (t_max - t_min) * (progress ** 2)
        else:
            temperature = t_min + (t_max - t_min) * progress
        
        temperature = float(max(temperature, 1e-6))

        if cfs_scheduler is not None:
            current_cfs = cfs_scheduler.get_cfs(step, num_steps)
        else:
            current_cfs = cfs

        t = torch.full((x_t.shape[0],), step * dt, device=device, dtype=torch.float32)
        need_attn = (return_attn_list and step == num_steps - 1)

        if use_attention:
            if need_attn:
                conditional_v, attn_list = flow_matcher(
                    x_t, t, y_class=conditional_class, y_activity=conditional_activity,
                    attention_mask=attention_mask, return_attn_matrix=True
                )
            else:
                conditional_v = flow_matcher(
                    x_t, t, y_class=conditional_class, y_activity=conditional_activity,
                    attention_mask=attention_mask
                )
            unconditional_v = flow_matcher(
                x_t, t, y_class=unconditional_class, y_activity=unconditional_activity,
                attention_mask=attention_mask
            )
        else:
            conditional_v = flow_matcher(x_t, t, y_class=conditional_class, y_activity=conditional_activity)
            unconditional_v = flow_matcher(x_t, t, y_class=unconditional_class, y_activity=unconditional_activity)

        pred_v = (1.0 + current_cfs) * conditional_v - current_cfs * unconditional_v
        
        pred_v = pred_v / temperature

        if clamp_value is not None and clamp_value > 0:
            pred_v = torch.clamp(pred_v, -abs(clamp_value), abs(clamp_value))

        if use_midpoint:
            x_mid = x_t + 0.5 * dt * pred_v
            t_mid = torch.full_like(t, (step + 0.5) * dt)
            midpoint_step = min(step + 0.5, max(0, num_steps - 1))
            midpoint_progress = float(midpoint_step) / float(max(1, num_steps - 1))

            if temp_schedule == 'linear':
                midpoint_temperature = t_min + (t_max - t_min) * midpoint_progress
            elif temp_schedule == 'cosine':
                midpoint_temperature = t_min + (t_max - t_min) * (
                    0.5 - 0.5 * math.cos(math.pi * midpoint_progress)
                )
            elif temp_schedule == 'quadratic':
                midpoint_temperature = t_min + (t_max - t_min) * (midpoint_progress ** 2)
            else:
                midpoint_temperature = t_min + (t_max - t_min) * midpoint_progress

            midpoint_temperature = float(max(midpoint_temperature, 1e-6))

            if cfs_scheduler is not None:
                midpoint_cfs = cfs_scheduler.get_cfs(midpoint_step, num_steps)
            else:
                midpoint_cfs = cfs
            
            if use_attention:
                need_attn_mid = (return_attn_list and step == num_steps - 1)
                if need_attn_mid:
                    conditional_v_mid, attn_list = flow_matcher(
                        x_mid, t_mid, y_class=conditional_class, y_activity=conditional_activity,
                        attention_mask=attention_mask, return_attn_matrix=True
                    )
                else:
                    conditional_v_mid = flow_matcher(
                        x_mid, t_mid, y_class=conditional_class, y_activity=conditional_activity,
                        attention_mask=attention_mask
                    )
                unconditional_v_mid = flow_matcher(
                    x_mid, t_mid, y_class=unconditional_class, y_activity=unconditional_activity,
                    attention_mask=attention_mask
                )
            else:
                conditional_v_mid = flow_matcher(x_mid, t_mid, y_class=conditional_class, y_activity=conditional_activity)
                unconditional_v_mid = flow_matcher(x_mid, t_mid, y_class=unconditional_class, y_activity=unconditional_activity)

            pred_v_mid = (1.0 + midpoint_cfs) * conditional_v_mid - midpoint_cfs * unconditional_v_mid
            pred_v_mid = pred_v_mid / midpoint_temperature
            
            if clamp_value is not None and clamp_value > 0:
                pred_v_mid = torch.clamp(pred_v_mid, -abs(clamp_value), abs(clamp_value))

            x_t = x_t + dt * pred_v_mid
        else:
            x_t = x_t + dt * pred_v
            
    return (x_t, attn_list) if return_attn_list else x_t

flow_mlp = [int(i) for i in conf_dict['denoiser_mlp'].split(',')]
flow_matcher = FlowMatcher(
    conf_dict['denoiser_esm_model_name'],
    int(conf_dict['denoiser_embedding']),
    flow_mlp
).cuda()
flow_matcher.load_state_dict(torch.load("save_model/flowy_matcher_model.pkl"))

tokenizer = AutoTokenizer.from_pretrained(conf_dict['denoiser_esm_model_name'], trust_remote_code=True)
esm2_model = AutoModelForMaskedLM.from_pretrained(conf_dict['denoiser_esm_model_name'], trust_remote_code=True).cuda()
decoder = esm2_model.lm_head

flow_matcher.eval()
esm2_model.eval()
decoder.eval()

if sample_type == 'amp':
    peptide_file = 'antimicrobial'
    decoder.load_state_dict(torch.load("save_model/antimicrobial_decoder_model_1.pkl"))
elif sample_type == 'afp':
    peptide_file = 'antifungal'
    decoder.load_state_dict(torch.load("save_model/antifungal_decoder_model_1.pkl"))
else:
    peptide_file = 'antiviral'
    decoder.load_state_dict(torch.load("save_model/antiviral_decoder_model_1.pkl"))


cls_id = getattr(tokenizer, "cls_token_id", 0)
eos_id = getattr(tokenizer, "eos_token_id", 2)
pad_id = getattr(tokenizer, "pad_token_id", 1)
mask_id = getattr(tokenizer, "mask_token_id", None)
illegal_token_ids = {3, 29, 30, 31} - {cls_id, eos_id, pad_id}
vocab_dict = {v: k for k, v in tokenizer.get_vocab().items()}

def decode_and_filter(seq_ids, seen_set):
    if cls_id is not None and seq_ids[0].item() != cls_id:
        return None, "no_cls"
    if any((seq_ids == v).any().item() for v in illegal_token_ids):
        return None, "illegal_token"
    
    eos_pos = (seq_ids == eos_id).nonzero(as_tuple=True)[0]
    if eos_pos.numel() == 0:
        return None, "no_eos"
    
    eos_index = int(eos_pos[0].item())
    if eos_index <= 1:
        return None, "too_short"
        
    seq = tokenizer.decode(seq_ids[1:eos_index], skip_special_tokens=True).replace(" ", "")
    
    if len(seq) < 3:
        return None, "short"
        
    if 'X' in seq:
        seq = ''.join(vocab_dict[random.randrange(4, 24)] if aa == 'X' else aa for aa in seq)
        
    if seq in seen_set:
        return None, "duplicate"
        
    seen_set.add(seq)
    return seq, "ok"

target_total = sample_num * n
BATCH_MAX = 1000
seq_list = []
seen_set = set()

with torch.no_grad():
    for round_i in range(10000):
        remaining = target_total - len(seq_list)
        if remaining <= 0:
            break
            
        batch_size = min(BATCH_MAX, remaining)
        
        x1 = flow_matching_sample(
            flow_matcher,
            [batch_size, int(conf_dict['max_length']) + 2, int(conf_dict['denoiser_embedding'])],
            num_steps=200,
            cfs=cfs,
            use_attention=True,
            peptide_type=peptide_file,
            index=(round_i + 1),
            class_label=labels[peptide_file],
            activity_label=1,
            t_min=t_min,
            t_max=t_max,
            dynamic_cfs=USE_DYNAMIC_CFS,
            cfs_schedule_type=CFS_SCHEDULE,
            cfs_min=CFS_MIN,
            cfs_max=CFS_MAX
        )
        
        pred_score = decoder(x1)
        seq_ids_list = pred_score.argmax(dim=-1)
        
        counts = defaultdict(int)
        for seq_ids in seq_ids_list:
            seq, reason = decode_and_filter(seq_ids, seen_set)
            counts[reason] += 1
            if seq:
                seq_list.append(seq)
        
        print(f"[Round {round_i+1}] Batch={batch_size} | Accepted={counts['ok']} | "
              f"Total={len(seq_list)}/{target_total}")
        
        if round_i > 200 and len(seq_list) == 0:
            raise RuntimeError("No valid sequences produced after many rounds. Check decoder/tokenizer.")

if len(seq_list) == 0:
    raise RuntimeError("No valid sequences generated.")

seq_list = [s for s in seq_list if s and len(s.strip()) > 0]
seq_list = list(dict.fromkeys(seq_list))

if len(seq_list) > sample_num:
    seq_list = seq_list[:sample_num]
elif len(seq_list) < sample_num:
    print(f"[WARN] Only generated {len(seq_list)}/{sample_num} valid sequences")

print(f"\n{'='*60}")
print("Evaluating metrics...")
print(f"{'='*60}")

pseudo_perplexity_list = []
entropy_list = []

batch_size = 8
for start in range(0, len(seq_list), batch_size):
    batch_seqs = seq_list[start:start + batch_size]
    batch_pp = []
    
    for seq in batch_seqs:
        if 'X' in seq:
            seq = ''.join(
                vocab_dict[random.randrange(4, 24)] if aa == 'X' else aa for aa in seq
            )
            
        tensor_input = tokenizer.encode(seq, return_tensors='pt')
        if tensor_input.size(-1) <= 3:
            continue
            
        repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
        labels_tensor = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100).to(device)
        
        try:
            with torch.cuda.amp.autocast(), torch.no_grad():
                loss = esm2_model(masked_input.to(device), labels=labels_tensor).loss
                pseudo_perplexity = loss.exp().detach().cpu()
                batch_pp.append(pseudo_perplexity)
        except torch.cuda.OutOfMemoryError:
            print("[WARN] OOM on sequence, skipping...")
            torch.cuda.empty_cache()
            continue
            
    pseudo_perplexity_list.extend(batch_pp)
    
    for seq in batch_seqs:
        entropy_dic = Counter(seq)
        entropy = -sum((v / len(seq)) * math.log2(v / len(seq)) for v in entropy_dic.values())
        entropy_list.append(entropy)

pseudo_perplexity_tensor = torch.tensor(pseudo_perplexity_list)
entropy_tensor = torch.tensor(entropy_list)

print("Pseudo-Perplexity: %.4f " % (pseudo_perplexity_tensor.mean().item()))
print("Entropy: %.4f " % (entropy_tensor.mean().item()))

def match_score(sample_seq, real_file):
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    score_list = []
    for record in SeqIO.parse(real_file, "fasta"):
        ref_seq = str(record.seq)
        alignments = aligner.align(ref_seq, sample_seq)
        score = alignments.score
        score_list.append(score)
    return np.mean(score_list)

real_file = f"data/train/{peptide_file}.fasta"
similarity_scores = []

print(f"\n[INFO] Calculating similarity scores with {real_file} ...")
for seq in tqdm(seq_list, desc="Calculating similarity"):
    try:
        score = match_score(seq, real_file)
        similarity_scores.append(score)
    except Exception as e:
        print(f"[WARN] failed to align sequence: {seq[:10]}... ({e})")
        continue

if len(similarity_scores) > 0:
    similarity_mean = np.mean(similarity_scores)
    similarity_std = np.std(similarity_scores)
    print("Similarity: %.4f " % (similarity_mean))
else:
    print("[WARN] No valid sequences for similarity computation.")

instability_scores = []
for seq in seq_list:
    try:
        analysis = ProteinAnalysis(seq)
        instability_scores.append(analysis.instability_index())
    except Exception:
        pass

if instability_scores:
    print("Instability: %.4f" % (mean(instability_scores)))

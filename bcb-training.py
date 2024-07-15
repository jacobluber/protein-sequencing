# %%
from torch.utils.data import Dataset
# from peft import get_peft_model, LoraConfig
import re, torch
from Bio import SeqIO
from transformers import BertTokenizer, BertForMaskedLM

TRAIN_SIZE = 50_000
EVAL_SIZE = 25_000

data_dir = "/home/tlp5359/data-UniRef/E-Coli/no-HP/"
output_dir = "models/E-Coli-FFT/KCYMR"
train_data_path  = f"{data_dir}/E-Coli_UniRef100-TRAINING.fasta"
eval_data_path  = f"{data_dir}/E-Coli_UniRef100-EVAL.fasta"

# ------ 
model_name = "Rostlab/prot_bert"
masking_letters = r"[AQNGDEILFPVHWST]" #r"[ARQNGHILMFPSTWV]"
max_length = 1024
ignore_list = ["[PAD]", "X"]

# ---
aa_replacements = {"U": "X", "Z": "X", "O": "X", "B": "X"} 


# %%
class ProteinDataset(Dataset):
    def __init__(self, sequences, 
                 tokenizer, tokenizer_kwargs = {'return_tensors':'pt', 
                                                'max_length':1024, 
                                                'padding':'max_length', 
                                                'truncation':True},
                 masking_letters=r"[GIFTPVAL]", 
                 sep=" ", mask = "[MASK]", ignore_list = None,
                 precompute_encodings = True):

        # load all sequences and filter
        vocab = [a for a in masking_letters if a.isalpha()]
        self.sequences = [ s for s in sequences if any(a in s for a in vocab)] # ignore no-mask sequences
        print("init # seq =",len(sequences))
        if len(sequences) != len(self.sequences):
            print("From ProteinDataset: Number of qualified sequences: ",len(self.sequences))
            print("From ProteinDataset: Rejected sequences:", vocab)
            print('\n'.join([ s for s in sequences if all(a not in s for a in vocab)]))
        
        self.masking_letters = masking_letters
        self.sep = sep
        self.mask = mask
        self.tokenizer = tokenizer
        self.ignore_list = ignore_list
        self.tokenizer_kwargs = tokenizer_kwargs
        self.precompute_encodings = precompute_encodings
        if precompute_encodings:
            self.encodings = self.tokenize(sequences,masking_letters)
        
    def tokenize(self, sequences, masking_letters):
        if isinstance(sequences,list):
            seqs = [(self.sep).join(s) for s in sequences]
            mseqs = [re.sub(masking_letters, self.mask, seq) for seq in seqs]
            encodings = self.tokenizer(text = mseqs, 
                                    text_target = seqs, **self.tokenizer_kwargs)
            labels = encodings['labels']
            ignore_list = self.ignore_list
            if ignore_list is not None:
                for elem in ignore_list:
                    elem_id = self.tokenizer.vocab[elem]
                    labels[labels == elem_id] = -100
            return encodings
        elif isinstance(sequences,str):
            seq = sequences
            encodings = self.tokenizer(text = re.sub(masking_letters, self.mask, seq), 
                                    text_target = seq, **self.tokenizer_kwargs)
            labels = encodings['labels']
            ignore_list = self.ignore_list
            if ignore_list is not None:
                for elem in ignore_list:
                    elem_id = self.tokenizer.vocab[elem]
                    labels[labels == elem_id] = -100
            return { k: v[0].clone().detach() for k,v in encodings.items() }
        else:
            print("error in ProteinDataset.__getitem__: sequences is None")
            exit(1)
                
    def __getitem__(self, idx):
        if self.precompute_encodings:
            return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        else:
            return self.tokenize(self.sequences[idx], masking_letters)

            
    def __len__(self):
        return len(self.sequences)
 

# %%
# INIT
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
tokenizer_kwargs = {'return_tensors':'pt', 'max_length': max_length, 'padding':'max_length', 'truncation':True}


# %%
# PARSING
translation_table = str.maketrans(aa_replacements)
vocab = [char for char in masking_letters if char.isalpha()]
train_sequences = [ str(r.seq).translate(translation_table) for r in SeqIO.parse(train_data_path, "fasta") if any(a in r.seq for a in vocab) ][ :TRAIN_SIZE]
eval_sequences =  [ str(r.seq).translate(translation_table) for r in SeqIO.parse(eval_data_path, "fasta") if any(a in r.seq for a in vocab)  ][ : EVAL_SIZE]

print('max_length: ', max_length)

print(f"------ training on {len(train_sequences)} sequences")
print(f"------ validate on {len(eval_sequences)} sequences")
print(f"output: {output_dir}")

# %%
# CONSTRUCTING DATASET
train_dataset = ProteinDataset(train_sequences,tokenizer, tokenizer_kwargs, masking_letters=masking_letters, ignore_list = ignore_list)
eval_dataset  = ProteinDataset( eval_sequences,tokenizer, tokenizer_kwargs, masking_letters=masking_letters, ignore_list = ignore_list)
print("DATASET's masking letters = ",train_dataset.masking_letters)
# %%
total_params = sum(p.numel() for p in model.parameters()) 
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:3,.0f}\nTrainable parameters: {trainable_params:3,.0f}")

# %%
# EVALUATION METRICS
# import evaluate
from torch.nn import CrossEntropyLoss
only_masked_tokens = [ tokenizer.vocab[c] for c in tokenizer.vocab.keys() if c not in masking_letters ]
loss_fct = CrossEntropyLoss(reduction='mean')
# Initialize global accumulators
global_accuracy = []
global_accuracy_masked = []
global_loss = []
def compute_metrics(eval_pred, compute_result=False):
    logits, labels = eval_pred
    # Ensure logits and labels are tensors
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    # Move logits and labels to the same device
    device = logits.device
    labels = labels.to(device)
    
    predictions = torch.argmax(logits, axis=-1)
    # all token acc (except ignored tokens)
    ignored_eval_tokens = torch.tensor([-100], device=device)
    ignored_mask = ~torch.isin(labels.view(-1), ignored_eval_tokens)
    batch_accuracy = (torch.sum(predictions.view(-1)[ignored_mask] == labels.view(-1)[ignored_mask])/torch.sum(ignored_mask)).item()
    # only masked token acc
    ignored_eval_tokens = torch.tensor(only_masked_tokens + [-100], device=device)
    ignored_mask = ~torch.isin(labels.view(-1), ignored_eval_tokens)
    batch_accuracy_m = (torch.sum(predictions.view(-1)[ignored_mask] == labels.view(-1)[ignored_mask])/torch.sum(ignored_mask)).item()
    # eval loss
    batch_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).item()
    
    # Accumulate statistics
    global_accuracy.append(batch_accuracy)
    global_accuracy_masked.append(batch_accuracy_m)
    global_loss.append(batch_loss)
    
    if compute_result:
        # Compute global summary statistics
        mean_accuracy = sum(global_accuracy) / len(global_accuracy)
        mean_accuracy_m = sum(global_accuracy_masked) / len(global_accuracy_masked)
        mean_loss = sum(global_loss) / len(global_loss)
        # Clear accumulators for the next evaluation
        global_accuracy.clear()
        global_accuracy_masked.clear()
        global_loss.clear()
        out = {'accuracy_per_token': mean_accuracy,
               'accuracy_per_masked_token': mean_accuracy_m, 
               'eval_loss': mean_loss}
        return out
    else:
        out = {'accuracy_per_token': batch_accuracy,
               'accuracy_per_masked_token': batch_accuracy_m,
               'eval_loss': batch_loss}
        return out
    
# %%

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
training_args = TrainingArguments(
  output_dir=output_dir,
  eval_strategy="steps",
  logging_dir=f"{output_dir}/logs",  # Directory to save logs
  save_total_limit = 20, 
  num_train_epochs = 200,
  logging_steps = 500, 
  save_steps= 500,
  per_device_train_batch_size=50,  # 16
  per_device_eval_batch_size=50,   
  load_best_model_at_end=True, 
  fp16=True,
  learning_rate = 0.00005,
  metric_for_best_model = "eval_loss",
  label_names=["labels"],
#   batch_eval_metrics=True  # Enable batch evaluation metrics
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=eval_dataset,
  compute_metrics=compute_metrics,
  callbacks = [EarlyStoppingCallback(early_stopping_patience=19, 
                                     early_stopping_threshold = 0.01,
                                     )],
  # tokenizer = tokenizer,
  # aa_weights = aa_weights,
  # device = device
)

# %%
# trainer.train()
trainer.train(resume_from_checkpoint="/home/tlp5359/projects/CC_Sequencing_LLM/BCB/models/E-Coli-FFT/KCYMR/checkpoint-16500")
# %%
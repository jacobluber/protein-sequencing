# THIS HAS LOOPING THRU INPUT_ARGS + SCOREBOARD
# %%
from transformers import BertTokenizer, BertForMaskedLM, pipeline
from Bio import SeqIO
import torch, re, time, random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
# import pandas as pd
import json,  os
# import dill 

# %%
model_path  = "/home/tlp5359/projects/CC_Sequencing_LLM/BCB/models/"

test_data_paths = {
    "E-albertii" : "/home/tlp5359/data-UniRef/E-albertii/no-HP/E-albertii_UniRef100-TESTING.fasta",
    "E-Coli"     : "/home/tlp5359/data-UniRef/E-Coli/no-HP/E-Coli_UniRef100-TESTING.fasta",
    "E-fergusonii" : "/home/tlp5359/data-UniRef/E-fergusonii/no-HP/E-fergusonii_UniRef100-TESTING.fasta",
    }

TEST_SIZE = 5_000 
N = 3
random.seed(3333)

test_set = []
masking_letters = r"[AQNGDEILFPV]"
known_aas = f"[{''.join([ a for a in 'AQNGDEILFPVRHWSTKCYM' if a not in masking_letters ])}]" # == knowns aa
aa_replacements = {"U": "X", "Z": "X", "O": "X", "B": "X"} 

vocab = [char for char in masking_letters if char.isalpha()]

max_length_filter = 5_000 # ~7000 will not fit 1 gpu 

for n, in_path in test_data_paths.items():
    records = []
    for record in SeqIO.parse(in_path, "fasta"):
        if any(a in record.seq for a in vocab) and len(record.seq) <= max_length_filter:
            records.append(record)
    sampling_records = random.sample(records, TEST_SIZE)
    test_set += [{"label": n, "record":record} for record in sampling_records]
            
        
print(f"------ total of {len(test_set)} fasta records")
print("max length from samples:", max([len(x['record'].seq) for x in test_set]))
# %%
input_args=[
    # {
    #     'model_path' : f"/home/tlp5359/projects/CC_Sequencing_LLM/BCB/models/E-Coli-FFT/KCYMRHWST/checkpoint-14500",
    #     'outname' : f"inference-results/KCYMRHWST/E-Coli-to-others",
    # },    
    
    {
        'model_path' : f"/home/tlp5359/projects/CC_Sequencing_LLM/BCB/models/E-albertii-FFT/KCYMRHWST/checkpoint-13500",
        'outname' : f"inference-results/KCYMRHWST/E-albertii-to-others",
    },    
    
    # {
    #     'model_path' : f"/home/tlp5359/projects/CC_Sequencing_LLM/BCB/models/E-fergusonii-FFT/KCYMRHWST/checkpoint-23500",
    #     'outname' : f"inference-results/KCYMRHWST/E-fergusonii-to-others",
    # },   
    
    # ---------------------------------------------------------------
    # {
    #     'model_path' : f"/home/tlp5359/projects/CC_Sequencing_LLM/BCB/models/E-Coli-FFT/KCYMRHWST/checkpoint-9000",
    #     'outname' : f"inference-results/KCYMRHWST/E-Coli-to-others-cp9000",
    # },    
    ]

# %%
from torch.utils.data import Dataset
class ProteinTestingDataset(Dataset):
    def __init__(self, sequences, masking_letters=r"[GIFTPVAL]", sep=" ", mask = "[MASK]"):
        vocab = [a for a in masking_letters if a.isalpha()]
        self.sequences = [ s for s in sequences if any(a in s for a in vocab)]
        
        if len(sequences) != len(self.sequences):
            print("From ProteinTestingDataset: Number of qualified sequences: ",len(self.sequences))
            print("From ProteinTestingDataset: Rejected sequences:", vocab)
            print('\n'.join([ s for s in sequences if all(a not in s for a in vocab)]))
        
        self.masking_letters = masking_letters
        self.sep = sep
        self.mask = mask
        
    def __getitem__(self, idx):
        sep = self.sep
        masking_letters =self.masking_letters
        mask = self.mask
        sequence = self.sequences[idx]

        seq = sep.join(sequence)
        masked_seq = re.sub(masking_letters, mask, seq)
        return masked_seq
    def __len__(self):
        return len(self.sequences)


def get_scoreboard(out):
  if isinstance(out[0],dict): # only 1 mask in seq, hence list is unwrapped
    return [{k['token_str']: k['score'] for k in out}]
  else:
    return [{k['token_str']: k['score'] for k in m} for m in out]
  
  
def string_matching_percentage(istr1, istr2, merge_dict={"[QN]":"Q"},ignore_lst="[KCYDE]", default=None):
    str1 = istr1
    str2 = istr2
    if ignore_lst is not None:
        # masking KCYDE
        str1 = re.sub(ignore_lst,'',str1)
        str2 = re.sub(ignore_lst,'',str2)
    if merge_dict is not None:
        # merging
        for k,v in merge_dict.items():
            str1 = re.sub(k,v,str1)
            str2 = re.sub(k,v,str2)
    # calculating matching score
    if len(str1) != len(str2):
        if default is None:
            print("string 1:", str1,"\nstring 2:", str2)
            raise ValueError("Both strings must have the same length for comparison. (You can set default (float) = something to bypass this error)")
        else:
            return default      
    matching_count = sum(1 for a, b in zip(str1, str2) if a == b and a.isalpha() )
    total_length = len(str1)
    
    percentage = (matching_count / total_length) * 100.0
    return percentage
  

def insert_space_for_gaps(expected, predicted):
    str1_parts = expected.split()
    str2_parts = []

    for part in str1_parts:
        str2_parts.append(predicted[:len(part)])
        predicted = predicted[len(part):]

    return ' '.join(str2_parts)

def get_one_token(lst):
        return lst['token_str'] if isinstance(lst, dict) else lst[0]['token_str']

# Function to plot the distribution of line lengths
def plot_line_length_distribution(line_lengths,bins=200):
    plt.figure(figsize=(14, 4))
    n, bins, patches = plt.hist(line_lengths, bins=bins, color='skyblue', edgecolor='black')
    plt.title('Distribution of Line Lengths')
    plt.xlabel('Line Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    print("Bin size:",bins[2]-bins[1])
    plt.show()

# %%
for arg_list in input_args:
    
    model_path = arg_list['model_path']
    outname = arg_list['outname']

    st =time.time()

    os.system(f'mkdir -p {outname}')
    outname += f"/UniRef100"
    

    # %%
    batch_dicts = [
                
                {'min_length':     0, 'max_length':   250, 'batch_size':    32},
                {'min_length':   250, 'max_length':   500, 'batch_size':    16},
                {'min_length':   500, 'max_length':   750, 'batch_size':     8},
                {'min_length':   750, 'max_length': 1_000, 'batch_size':     6},
                {'min_length': 1_000, 'max_length': 2_000, 'batch_size':     3},                
                {'min_length': 2_000, 'max_length': 5_000, 'batch_size':     1},
                {'min_length': 5_000, 'max_length': 100000000000000000000000, 'batch_size':1},                
                ]

    # %%
    torch.cuda.empty_cache()
    
    # LOAD FINETUNED MODEL
    model = BertForMaskedLM.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
    pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer, device='cuda', targets=vocab, top_k=len(vocab))
    
    output = {'seq_label': [], 'original_20aas': [], 'original sequence': [], 'description': [], 
                'seq_length': [], 'Masking %': [], 'Matching Percentage': [], 'Matching Percentage full': [], 
                'full predicted': [], 'expected': [], 'predicted': [],'scoreboard': []}
    
    total_st = time.time()
    
    print("#"*30,f"\nPERFORM FOLD #{N} - {model_path.split('/')[-2]}")
    for bd in batch_dicts:
        start_time = time.time()
        testing_dataset = None
        min_length = bd['min_length']
        max_length = bd['max_length']
        batch_size = bd['batch_size']
        test_subset = [ r for r in test_set if len(r['record'].seq) <= max_length and len(r['record'].seq) > min_length ]
        testing_records = [ r["record"] for r in test_subset ]
        seq_labels = [ r["label"] for r in test_subset ]
        
        testing_sequences_20aas = [ str(r.seq) for r in testing_records]
        if aa_replacements is not None:
            translation_table = str.maketrans(aa_replacements)
            testing_sequences = [ s.translate(translation_table) for s in testing_sequences_20aas] 
        else:
            testing_sequences = testing_sequences_20aas 
        
        record_descriptions = [ r.description for r in testing_records]
        if testing_sequences != []:
            testing_sequences_length = [len(x) for x in testing_sequences]
            masking_percentage = [sum(1 for char in s if char in masking_letters)/len(s) * 100 for s in testing_sequences]
            not_masking_letters=''.join(("[^",masking_letters[1:]))
            expected_masks = [" ".join(re.sub(not_masking_letters," ",s).split()) for s in testing_sequences]
            
            print(f"for {min_length} < length <= {max_length}; batch_size = {batch_size}: unmasking {len(testing_sequences)} sequence(s)")

            # Construct testing dataset
            testing_dataset = ProteinTestingDataset(testing_sequences, masking_letters= masking_letters)

            # unmasking
            mask_results = []
            score_tables = []
            for out in tqdm(pipe(testing_dataset, batch_size=batch_size), total=len(testing_dataset)):
                score_tables.append(get_scoreboard(out))
                mask_results.append("".join([get_one_token(item) for item in out]))
                pass
            
            full_predicted_sequences = []
            for i,s in enumerate(testing_sequences):
                full_predicted = ""
                counter = 0
                for o in s:
                    if o in masking_letters:
                        full_predicted += mask_results[i][counter]
                        counter +=1
                    else:
                        full_predicted +=o
        
                full_predicted_sequences.append(full_predicted)


            mask_results = [insert_space_for_gaps(e,p) for e,p in zip(expected_masks, mask_results)]

            matching_scores = [string_matching_percentage(e,p,merge_dict=None,ignore_lst=' ',default=-1) for e,p in zip(expected_masks, mask_results)]
            matching_scores_full = [string_matching_percentage(e,p,merge_dict=None,ignore_lst=None,default=-1) for e,p in zip(testing_sequences_20aas, full_predicted_sequences)]

            # outputing
            output['seq_label'] += seq_labels
            output['original_20aas'] += testing_sequences_20aas
            output['original sequence'] += testing_sequences
            output['description'] += record_descriptions
            output['seq_length'] += testing_sequences_length
            output['Masking %'] += masking_percentage
            output['Matching Percentage'] += matching_scores
            output['Matching Percentage full'] += matching_scores_full
            output['full predicted'] += full_predicted_sequences
            output['expected'] += expected_masks
            output['predicted'] += mask_results
            output['scoreboard'] += score_tables

            print("# seqs: ", len(testing_sequences_20aas))
            print("Current Ave. matching score: ",sum(output['Matching Percentage'])/len(output['Matching Percentage']))
            print("-------------Time executed: ", time.time()-start_time)
            

    print("Average Matching score: ",sum(output['Matching Percentage'])/len(output['Matching Percentage']))
    
    with open(f'{outname}-F{N}.json', "w") as json_file:
        json.dump(output, json_file, indent=4)
    # -------------------------
    print(f"TOTAL TIME for {len(test_set)} sequences: {time.time()-total_st}")

from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)

        
inputs = tokenizer("maybe The [MASK] of northern [MASK] is Paris.", return_tensors="pt").to(device)
labels = tokenizer("maybe The capital of northern France is Paris.", return_tensors="pt")["input_ids"].to(device)

outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits
probs = torch.softmax(logits, dim=-1)
output_seq = []
for p in probs.squeeze():
    p = p.detach().cpu().numpy()
    output_seq.append(np.argmax(p))
decoded = tokenizer.decode(output_seq, skip_special_tokens=True)
print(decoded)

inputs_list = inputs["input_ids"].cpu().flatten().numpy()
chosen_indices = np.random.choice(range(len(inputs_list)), size=int(len(inputs_list)*50/100))
print(chosen_indices)
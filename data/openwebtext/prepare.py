import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset 

# Number of processes for parallelization
num_proc = os.cpu_count()  # Use all available CPU cores
print("num_processes :", num_proc)

# Number of workers for loading the dataset
num_proc_load_dataset = num_proc

# Initialize the GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load the OpenWebText dataset
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # Split the dataset into training and validation sets
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=1234, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # Rename the test split to validation

    # Define a function to tokenize the text
    def process(example):
        ids = enc.encode_ordinary(example['text'])  # Tokenize the text
        ids.append(enc.eot_token)  # Append the end-of-text token
        return {'ids': ids, 'len': len(ids)}

    # Tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing the splits",
        num_proc=num_proc,
    )

    # Save the tokenized data to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # Use 16-bit integers for storage
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
            # Process data in batches for efficiency
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch  # Write batch to memory-mapped file
            idx += len(arr_batch)
        arr.flush()  # Ensure all data is written to disk

    # The resulting files:
    # - train.bin (~17GB) contains ~9B tokens
    # - val.bin (~8.5MB) contains ~4M tokens

    # To read the binary files later:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')

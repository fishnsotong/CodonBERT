# TODO: this is very naive, perhaps we can do some clustering and k-fold cv soon?

import argparse
import random

def read_fasta(file_path):
    """Reads a FASTA file and returns a list of (header, sequence) tuples."""
    sequences = []
    with open(file_path, "r") as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):  # New sequence header
                if header:
                    sequences.append((header, "".join(seq_lines)))  # Save previous sequence
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header:  # Save last sequence
            sequences.append((header, "".join(seq_lines)))
    return sequences

def write_fasta(file_path, sequences):
    """Writes a list of (header, sequence) tuples to a FASTA file."""
    with open(file_path, "w") as f:
        for header, seq in sequences:
            f.write(f"{header}\n{seq}\n")

def split_fasta(input_fasta, train_ratio, train_output, val_output, seed=42):
    """Splits a FASTA file into training and validation sets."""
    sequences = read_fasta(input_fasta)
    random.seed(seed)
    random.shuffle(sequences)  # Shuffle sequences before splitting

    split_idx = int(len(sequences) * train_ratio)
    train_set = sequences[:split_idx]
    val_set = sequences[split_idx:]

    write_fasta(train_output, train_set)
    write_fasta(val_output, val_set)

    print(f"Total sequences: {len(sequences)}")
    print(f"Training set: {len(train_set)} sequences -> {train_output}")
    print(f"Validation set: {len(val_set)} sequences -> {val_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a FASTA file into training and validation sets.")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-r", "--ratio", type=float, default=0.8, help="Train set ratio (default: 0.8)")
    parser.add_argument("-o_train", "--train_output", required=True, help="Output file for training set")
    parser.add_argument("-o_val", "--val_output", required=True, help="Output file for validation set")
    
    args = parser.parse_args()
    
    split_fasta(args.input, args.ratio, args.train_output, args.val_output)
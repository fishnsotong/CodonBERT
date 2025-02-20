import os
import logging
import requests
import zipfile
import gzip
import shutil
import pandas as pd

import subprocess, multiprocessing
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

tqdm.pandas()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

VERSION_HPA = 23
VERSION_GENCODE = 47

class GencodeLoader:
    # TODO: the constructor can be used to include certain things 
    def __init__(self, version_hpa: int, version_gencode: int, species: str = "human"):

        self.version_hpa = version_hpa
        self.version_gencode = version_gencode
        self.species = species
        self.data_dir = "data"

        # Ensure data is downloaded when object is instantiated
        self.paths = self._download_data()

    def _download_data(self):
        # URLs and target filenames
        files = [  # a list of dictionaries
            {
                "url": f"https://v{self.version_hpa}.proteinatlas.org/download/transcript_rna_tissue.tsv.zip",
                "path": "data/transcript_rna_tissue.tsv.zip",
                "extract": True,
                "extracted_path": "data/transcript_rna_tissue.tsv"
            },
            {
                "url": f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{self.version_gencode}/gencode.v{self.version_gencode}.pc_translations.fa.gz",
                "path": f"data/gencode.v{self.version_gencode}.pc_translations.fa.gz",
                "extract": True,
                "extracted_path": f"data/gencode.v{self.version_gencode}.pc_translations.fa"
            },
            {
                "url": f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{self.version_gencode}/gencode.v{self.version_gencode}.pc_transcripts.fa.gz",
                "path": f"data/gencode.v{self.version_gencode}.pc_transcripts.fa.gz",
                "extract": True,
                "extracted_path": f"data/gencode.v{self.version_gencode}.pc_transcripts.fa"
            },
            # we're not going to use the annotations file, but I'm leaving it here to download anyways because why not
            {
                "url": f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{self.version_gencode}/gencode.v{self.version_gencode}.annotation.gtf.gz",
                "path": f"data/gencode.v{self.version_gencode}.annotation.gtf.gz",
                "extract": True,
                "extracted_path": f"data/gencode.v{self.version_gencode}.annotation.gtf"
            }
        ]

        # Ensure data directory exists, create if it doesn't
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            logging.info(f"Created directory: {self.data_dir}")
        except PermissionError:
            logging.error(f"Permission denied when creating {self.data_dir}")
            raise
        except OSError as e:
            logging.error(f"Failed to create directory {self.data_dir}: {e}")
            raise

        # Download each file that we need
        for file in files:
            try:
                # Check if the file already exists
                if not os.path.exists(file["path"]) and not os.path.exists(file.get("extracted_path", "")):
                    logging.info(f"Downloading {file['url']}...")
                    response = requests.get(file["url"], stream=True)
                    response.raise_for_status()

                    with open(file["path"], "wb") as f:
                        shutil.copyfileobj(response.raw, f)
                    logging.info(f"Downloaded {file['path']}")
                else:
                    logging.info(f"{file['path']} exists, skipping download....")
                    continue

                # Extract the file if necessary
                if file.get("extract") and not os.path.exists(file["extracted_path"]):
                    if file["path"].endswith(".zip"):
                        logging.info(f"Extracting {file['path']}...")
                        with zipfile.ZipFile(file["path"], "r") as zip_ref:
                            zip_ref.extractall("data/")
                    elif file["path"].endswith(".gz"):
                        logging.info(f"Decompressing {file['path']}...")
                        with gzip.open(file["path"], "rb") as gz_ref:
                            with open(file["extracted_path"], "wb") as out_file:
                                shutil.copyfileobj(gz_ref, out_file)
                    logging.info(f"Extracted to {file['extracted_path']}")

                # Delete compressed download files
                os.remove(file["path"])

            # Error handling
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to download {file['url']}: {e}")
            except zipfile.BadZipFile:
                logging.error(f"Failed to extract zip file: {file['path']}")
            except OSError as e:
                logging.error(f"OS error during file handling: {e}")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")

        return [k["extracted_path"] for k in files]

    def _safe_get_coordinates(self, parts: list[str], prefix: str) -> tuple:
        """
        Safely extract coordinates for a given feature (UTR5, CDS, UTR3).
        Returns (start, stop) tuple if found, (None, None) if not present.
        """
        try:
            # Find the first matching element that starts with the prefix
            coordinates = next(a.split(":")[1].split("-") for a in parts[6:] if a.startswith(prefix))
            return coordinates[0], coordinates[1]
        
        except (StopIteration, IndexError):
            # Return None values if feature not found or parsing fails
            return None, None


    def load_hpa(self) -> tuple[pd.DataFrame, dict]:
        # See _download_data for which path to take for each dataset
        df = pd.read_csv(self.paths[0], sep="\t")

        logging.info(f"HPA dataframe size: {df.shape}")
        logging.info(f"Number of unique genes: {df['ensgid'].nunique()}")
        logging.info(f"Number of unique transcripts: {df['enstid'].nunique()}")

        # how many columns contain TPM information?
        tpm_columns = [col for col in df.columns if col.startswith('TPM.')]
        logging.info(f"Found {len(tpm_columns)} TPM columns: {tpm_columns[:2]}...")

        # The same for estimated counts (which is not used in this analysis)
        est_columns = [col for col in df.columns if col.startswith('est_counts.')]
        logging.info(f"Found {len(est_columns)} est_counts columns: {est_columns[:2]}...")

        # Create a list of different tissue types in the HPA dataset
        tissues = set([col.split(".")[1] for col in tpm_columns])
        logging.info(f"Found {len(tissues)} tissues")

        # Create metadata
        metadata = {
            "tpm_columns": tpm_columns,
            "est_columns": est_columns,
            "tissues": tissues,
        }
        return df, metadata

    def parse_gencode_translations_header(self, header: str) -> dict[str, str]:
        """
        Parse GENCODE translations FASTA header to extract relevant IDs. Below is the information present in a typical header:
        ENSP00000493376.2 - Ensembl Protein ID (with version 2)
        ENST00000641515.2 - Ensembl Transcript ID (with version 2)
        ENSG00000186092.7 - Ensembl Gene ID (with version 7)
        OTTHUMG00000001094.4 - HAVANA Gene ID (Old term: "OTTHUM" = human annotation from HAVANA team)
        OTTHUMT00000003223.4 - HAVANA Transcript ID
        OR4F5-201 - Gene name with transcript number (201 indicates this is transcript #1)
        OR4F5 - Gene symbol/name (in this case, Olfactory Receptor Family 4 Subfamily F Member 5)
        326 - Protein length in amino acids

        Example header:
        >ENSP00000493376.2|ENST00000641515.2|ENSG00000186092.7|OTTHUMG00000001094.4|OTTHUMT00000003223.4|OR4F5-201|OR4F5|326
        """
        parts = header.split('|')
        return {
            'protein_id': parts[0].strip('>').split('.')[0],
            'transcript_id': parts[1].strip('>').split('.')[0],
            'gene_id': parts[2].split('.')[0],
            'transcript_name': parts[5] if len(parts) > 5 else None,
            'gene_name': parts[6] if len(parts) > 6 else None
        }

    def parse_gencode_transcripts_header(self, header: str) -> dict[str, str]:
        """
        Parse GENCODE transcripts FASTA header to extract relevant IDs. Below is the information present in a typical header:
        ENST00000641515.2 - Ensembl Transcript ID (with version 2)
        ENSG00000186092.7 - Ensembl Gene ID (with version 7)
        OTTHUMG00000001094.4 - HAVANA Gene ID (Old term: "OTTHUM" = human annotation from HAVANA team)
        OTTHUMT00000003223.4 - HAVANA Transcript ID
        OR4F5-201 - Transcript name, specifying a particular transcript isoform for the OR4F5 gene. (201 indicates this is transcript #1)
        OR4F5 - Gene symbol/name (in this case, Olfactory Receptor Family 4 Subfamily F Member 5)
        2618 - Transcript length in nucleotides
        UTR5:1-60 - Nucleotide range within the transcript that constitutes the 5' UTR (from nucleotide 1 to 60).
        CDS:61-1041 - CDS is the region of the transcript that translates into protein. Indicates that nucleotides 61 to 1041 form the CDS.
        UTR3:1042-2618 - specifies that nucleotides 1042 to 2618 make up the 3' UTR.

        Example header:
        >ENST00000641515.2|ENSG00000186092.7|OTTHUMG00000001094.4|OTTHUMT00000003223.4|OR4F5-201|OR4F5|2618|UTR5:1-60|CDS:61-1041|UTR3:1042-2618|
        """
        parts = header.split('|')

        # 5UTR, CDS, 3UTR coordinates
        coords_5utr = self._safe_get_coordinates(parts, "UTR5")
        coords_cds = self._safe_get_coordinates(parts, "CDS")
        coords_3utr = self._safe_get_coordinates(parts, "UTR3")

        return {
            'transcript_id': parts[0].strip('>').split('.')[0],
            'gene_id': parts[1].split('.')[0],
            'transcript_name': parts[4] if len(parts) > 5 else None,
            'gene_name': parts[5] if len(parts) > 6 else None,
            '5utr_start': coords_5utr[0], '5utr_stop': coords_5utr[1],
            'cds_start': coords_cds[0], 'cds_stop': coords_cds[1],
            '3utr_start': coords_3utr[0], '3utr_stop': coords_3utr[1],
        }

    def load_gencode(self) -> tuple[dict[str, str], dict[str, str], dict[str, dict]]:
        """
        Load transcripts, protein sequences and their metadata from GENCODE FASTA.

        Returns:
            Tuple of (transcripts_dict, translations_dict metadata_dict)
        """
        transcripts = {}
        translations = {}
        metadata = {}

        # Load transcripts 
        with open(self.paths[2], "rt") as f:
            for record in SeqIO.parse(f, "fasta"):
                header_info = self.parse_gencode_transcripts_header(record.description)
                transcript_id = header_info['transcript_id']
                gene_id = header_info['gene_id']
                transcripts[gene_id + '_' + transcript_id] = str(record.seq)
                metadata[transcript_id] = header_info
        logging.info(f"Found {len(transcripts)} protein-coding RNA transcripts in GENCODE {self.version_gencode}")

        # Load protein sequences
        with open(self.paths[1], 'rt') as f:
            for record in SeqIO.parse(f, 'fasta'):
                header_info = self.parse_gencode_translations_header(record.description)
                transcript_id = header_info['transcript_id']
                gene_id = header_info['gene_id']
                translations[gene_id + '_' + transcript_id] = str(record.seq)
                metadata[transcript_id]["protein_id"] = header_info['protein_id']
        logging.info(f"Found {len(translations)} protein sequences in GENCODE {self.version_gencode}")
        return transcripts, translations, metadata

def tpm_filter(input_df: pd.DataFrame, tissues, tpm_columns, plotting=True) -> pd.DataFrame:
    enstid_all_tissues = []

    for tissue in tissues:
        # Retrieve columns for the current tissue
        tissue_cols = [col for col in tpm_columns if f'.{tissue}.' in col]

        # Create a new DataFrame with 'ensgid', 'enstid', and tissue columns from `hpa_df`
        tissue_df = input_df[['ensgid', 'enstid'] + tissue_cols].copy()

        # Calculate the median across tissue columns and store in a new 'TPM_median' column
        tissue_df.loc[:, 'TPM_median'] = tissue_df.loc[:, tissue_cols].median(axis=1)

        # Filter `tissue_df` to keep only rows where 'TPM_median' is 5 or more
        tissue_df = tissue_df[tissue_df['TPM_median'] >= 5]

        # Print the number of genes expressed in the current tissue with TPM >= 5
        # print(f"{tissue.capitalize()}: {tissue_df['ensgid'].count()} genes")

        # Store the enstid values in a list
        enstid_all_tissues.extend(tissue_df['enstid'].values)

    # Create a new DataFrame with only the unique transcripts
    tpm_filtered_df = hpa_df[hpa_df['enstid'].isin(set(enstid_all_tissues))]

    logging.info(f"Number of transcripts with median TPM >= 5 in"
                 f" at least one tissue: {tpm_filtered_df['enstid'].count()} transcripts")
    if plotting:
        # TODO: implement plotting logic
        pass

    return tpm_filtered_df

def dataframe_creation(
        DNA_seqs: dict[str, str],
        protein_seqs: dict[str, str],
        metadata: dict[str, dict],
) -> pd.DataFrame:
    DNA_CDS_seqs = {k: v[int(metadata[k.split("_")[1]]['cds_start']) - 1: int(metadata[k.split("_")[1]]['cds_stop'])]
                    for k, v in DNA_seqs.items()}
    
    df = pd.DataFrame({
        'enstid': [x.split("_")[1] for x in list(DNA_seqs.keys())],
        'gene_id': [x.split("_")[0] for x in list(DNA_seqs.keys())],
        'enspid': [metadata[k.split("_")[1]]['protein_id'] for k in DNA_seqs],
        'DNA_seq': list(DNA_seqs.values()),
        'protein_seq': list(protein_seqs.values()),
        'CDS_seq': list(DNA_CDS_seqs.values()),
        'gene_name': [metadata[k.split("_")[1]]['gene_name'] for k in DNA_seqs]
        })
    
    return df

# TODO: this function is kinda slow, can we fix this? (if it's shit, don't use it LOL)
# but it works! so keep it
def quality_control(
        DNA_seqs: dict[str, str],
        protein_seqs: dict[str, str],
        metadata: dict[str, dict],
        expression_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Match protein sequences with expression data.
    """
    # Track statistics
    stats = {
        'total_expression_records': len(expression_df),
        'matched_transcripts': 0,
        'unmatched_transcripts': 0,
        'multiple_matches': 0,
    }

    # Create a dictionary of coding DNA sequences (CDS) by slicing each DNA sequence (v) from the start to stop positions specified in the metadata.
    # Each key (k) in DNA_seqs is split to extract an identifier used to look up CDS start and stop positions in the metadata.
    # The slice is adjusted by subtracting 1 from the start to account for 0-based indexing.
    DNA_CDS_seqs = {k: v[int(metadata[k.split("_")[1]]['cds_start']) - 1: int(metadata[k.split("_")[1]]['cds_stop'])]
                    for k, v in DNA_seqs.items()}

    matched_data = pd.DataFrame(columns=['ensgid', 'enstid', 'prot_seq', 'DNA_seq'])

    # Process each expression record: iteration through filtered dataframe (TPM > 5)
    # iterrows allows us to iterate over the rows of a DataFrame in Pandas: row index, pd.Series containing the row’s data
    for _, row in expression_df.iterrows():
        idx = row['ensgid'] + '_' + row['enstid']
        if idx in protein_seqs.keys() and idx in DNA_CDS_seqs.keys():
            prot_id = row['ensgid']
            prot_seq = protein_seqs[idx]
            dna_id = row['enstid']
            dna_seq = DNA_CDS_seqs[idx]
        else:
            prot_id = None
            dna_id = None
            stats['unmatched_transcripts'] += 1
            continue

        if dna_id != None and dna_seq != None:

            if len(dna_seq) % 3 != 0:  # Check if len(CDS) isn't divisible by 3
                stats['unmatched_transcripts'] += 1
                continue
            else:  # This means len(CDS) % 3 == 0
                translated = str(Seq(dna_seq).translate(to_stop=False))
                if len(translated) == len(prot_seq):
                    if translated == prot_seq:  # or translated[1:] == prot_seq[1:]:
                        match_status = 1
                    else:
                        match_status = 0
                else:
                    # Handle cases where the translated sequence includes the STOP codon
                    if len(translated) - 1 == len(prot_seq):
                        # EITHER: remove just the STOP codon
                        # OR: remove both STOP codon and initial Met if everything else matches
                        if translated[:-1] == prot_seq:  # or translated[1:-1] == prot_seq[1:]:
                            match_status = 1
                    else:
                        match_status = 0

                # IF QUALITY CONTROL CHECKS ARE PASSED, CREATE A DATAFRAME ENTRY
                if match_status == 1:
                    matched_data.loc[len(matched_data)] = [row['ensgid'], row['enstid'],
                                                           str(translated.strip('[').strip(']').replace("\'", "")),
                                                           str(dna_seq.strip('[').strip(']').replace("\'", ""))]

                    stats['matched_transcripts'] += 1
                else:
                    stats['unmatched_transcripts'] += 1

    # Print matching statistics
    print("\nMatching Statistics:")
    print(f"Total expression records: {stats['total_expression_records']:,}")
    print(f"Matched transcripts: {stats['matched_transcripts']:,}")
    print(f"Unmatched transcripts: {stats['unmatched_transcripts']:,}")

    return matched_data

def fasta_to_dataframe(fasta_path: str) -> pd.DataFrame:
    """
    Reads a FASTA file and converts it into a pandas DataFrame.

    Args:
        fasta_path (str): Path to the FASTA file.

    Returns:
        pd.DataFrame: DataFrame with columns ['ensgid', 'DNA_seq'].
    """
    records = SeqIO.parse(fasta_path, "fasta")
    data = [{"enstid":  str(record.id),
            # "ensgid":  str(record.id).split("|")[0], "enstid":  str(record.id).split("|")[1], 
             "DNA_seq": str(record.seq)} for record in records]

    return pd.DataFrame(data)

def dataframe_to_fasta(df: pd.DataFrame, sequence_column: str, fasta_path: str) -> None:
    """
    Converts a pandas DataFrame into a FASTA file using Biopython.

    Args:
        df (pd.DataFrame): DataFrame with columns ['ensgid', 'DNA_seq'].
        sequence_column (str): Name of the column containing the primary sequence of the biomolecule.
        fasta_path (str): Path to save the FASTA file.
    """
    records = [SeqRecord(Seq(row[sequence_column]), id=row['enstid'], description="") for _, row in df.iterrows()]
    # records = [SeqRecord(Seq(row[sequence_column]), id=f"{row['ensgid']}|{row['enstid']}", description="") for _, row in df.iterrows()]
    SeqIO.write(records, fasta_path, "fasta-2line")
    logging.info(f"FASTA file saved as {fasta_path}")

def run_cai(fasta_path: str, cfile: str = "Ehuman.cut") -> pd.DataFrame:
    """
    Calculate the Codon Adaptation Index (CAI) for a set of sequences.

    Args:
        fasta_path (str): Path to the FASTA file containing sequences.
        cfile (str): Path to the codon frequency file.

    Returns:
        pd.DataFrame: DataFrame with columns ['ensgid', 'CAI'].
    """
    cai_results = []
    output_file = "cai.txt"
    
    try:
        output = subprocess.run(
            ["cai", "-seqall", fasta_path, "-cfile", cfile, "-outfile", output_file],
            text=True,
            capture_output=True,
            check=True
        )
        print(output.stdout) # some form of logging
        cai_results = {}
        df = fasta_to_dataframe(fasta_path)

        # parse output file generated by EMBOSS cai using string split
        with open(output_file, "r") as file:
            for line in file:
                parts = line.split()                             # splitting by whitespace
                if len(parts) == 4 and parts[0] == "Sequence:":  # ensure correct format
                    enstid = parts[1]
                    cai_value = float(parts[3])                  # extract CAI value
                    cai_results[enstid] = cai_value

        # convert to pandas Series and return a joined DataFrame
        cai_series = pd.Series(cai_results, name="CAI")
        return df.join(cai_series, on="enstid")
    
    except subprocess.CalledProcessError as e:
        print(f"EMBOSS cai encountered an error for {fasta_path}: {e}")
        return None
    
    except Exception as ex:
        print(f"An unexpected error occurred for {fasta_path}: {ex}")
        return None

    finally:
        # clean up temporary files
        if os.path.exists(output_file):
            os.remove(output_file)

def _run_cai(input_file, output_file, cfile: str = "Ehuman.cut"):
        """Run EMBOSS CAI on a given input file and save results."""
        try:
            output = subprocess.run(
                ["cai", "-seqall", input_file, "-cfile", cfile, "-outfile", output_file],
                text=True,
                capture_output=True,
                check=True
            )
            print(output.stdout)  # Logging CAI output
        except subprocess.CalledProcessError as e:
            print(f"Error running CAI for {input_file}: {e}")

def run_cai_multicore(fasta_path: str, cfile: str = "Ehuman.cut", num_processes: int = None) -> pd.DataFrame:
    """
    Calculate the Codon Adaptation Index (CAI) in parallel for a set of sequences.

    Args:
        fasta_path (str): Path to the FASTA file containing sequences.
        cfile (str): Path to the codon frequency file.
        num_processes (int, optional): Number of CPU cores to use. Defaults to all available cores.

    Returns:
        pd.DataFrame: DataFrame with columns ['ensgid', 'CAI'].
    """

    def split_fasta(input_fasta, num_splits):
        """Split a FASTA file into smaller chunks for parallel processing."""
        with open(input_fasta, "r") as f:
            lines = f.readlines()

        sequences = []
        current_seq = []
        for line in lines:
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                current_seq = [line]
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))

        num_seqs = len(sequences)
        chunk_size = max(1, num_seqs // num_splits)  # Ensure at least one sequence per chunk
        chunk_files = []

        for i in range(num_splits):
            chunk_file = f"split_{i}.fasta"
            chunk_files.append(chunk_file)
            with open(chunk_file, "w") as f:
                f.writelines(sequences[i * chunk_size : (i + 1) * chunk_size])

        return chunk_files

    def parse_cai_output(output_files):
        """Parse multiple CAI output files into a pandas DataFrame."""
        cai_results = {}

        for file in output_files:
            if not os.path.exists(file):
                continue  # Skip missing files
            with open(file, "r") as infile:
                for line in infile:
                    parts = line.split()
                    if len(parts) == 4 and parts[0] == "Sequence:":  # Ensure correct format
                        ensgid = parts[1]
                        cai_value = float(parts[3])
                        cai_results[ensgid] = cai_value

        return pd.Series(cai_results, name="CAI")

    # Determine number of processes to use
    if num_processes is None:
        num_processes = os.cpu_count()

    # Split the FASTA file
    chunk_files = split_fasta(fasta_path, num_processes)
    output_files = [f"output_{i}.txt" for i in range(num_processes)]

    # Run CAI in parallel
    with multiprocessing.Pool(num_processes) as pool:
        pool.starmap(_run_cai, zip(chunk_files, output_files))

    # Parse output and join with the original FASTA dataframe
    df = fasta_to_dataframe(fasta_path)
    cai_series = parse_cai_output(output_files)
    df = df.join(cai_series, on="ensgid")

    # Cleanup temporary files
    for file in chunk_files + output_files:
        if os.path.exists(file):
            os.remove(file)

    return df

def run_rnafold(sequence: str, name="sequence") -> float:
    """
    Runs RNAfold on a single sequence and extracts the minimum free energy (MFE).
    """
    try:
        sequence = sequence.replace("T", "U")
        result = subprocess.run(
            ["RNAfold", "--noPS"],              # command is case-sensitive in Linux systems, take note!
            input=f">{name}\n{sequence}\n",
            text=True,
            capture_output=True,
            check=True
        )
        output_lines = result.stdout.split("\n")
        for line in output_lines:
            if "(" in line:                      # Valid RNAfold output should contain "(" for structure
                mfe = line.split(" ")[-1]        # Extract the last part containing the MFE
                return float(mfe.strip("()"))    # Remove parentheses
    except subprocess.CalledProcessError as e:
        print(f"RNAfold encountered an error for {name}: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred for {name}: {ex}")
    return None

def parallel_mfe(df: pd.DataFrame, threads: int = 4) -> pd.DataFrame:
    """
    Computes the minimum free energy (MFE) of RNA sequences in parallel.

    This function processes RNA sequences from a DataFrame using the `run_rnafold` function.
    If `threads` is set to 1, it applies the function sequentially using `.apply()`. Otherwise,
    it utilizes `ThreadPoolExecutor` to process the sequences in parallel.

    Args:
        df (pd.DataFrame): A DataFrame containing RNA sequences. Must have at least two columns:
            - 'DNA_seq': The RNA sequence (assumed to be DNA before conversion).
            - 'ensgid': A unique identifier for each sequence.
        threads (int, optional): The number of parallel threads to use. Defaults to 4.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column, 'MFE', containing
        the computed minimum free energy values for each sequence.

    Notes:
        - If `threads` is set to 1, the function avoids the overhead of multithreading
          by using `.apply()`.
        - If `threads` > 1, `ThreadPoolExecutor` is used to execute `run_rnafold` in parallel,
          with a progress bar displayed using `tqdm`.

    """
    if threads == 1:
        # using .apply() instead of ThreadPoolExecutor to avoid unnecessary overhead
        df["MFE"] = df.apply(lambda row: run_rnafold(row.DNA_seq, name=row.enstid), axis=1)
        return df
    
    else:
        # Apply parallel execution
        with ThreadPoolExecutor(max_workers=threads) as executor:
        
            results = list(tqdm(
                executor.map(lambda row: run_rnafold(row.DNA_seq, name=row.enstid), df.itertuples(index=False)),
                total=len(df)
                ))
    
    df.loc[:, "MFE"] = results
    return df

if __name__ == '__main__':

    loader = GencodeLoader(VERSION_HPA, VERSION_GENCODE)

    # (1) LOAD (both expression data and sequences)
    hpa_df, hpa_metadata = loader.load_hpa()
    gencode_transcripts, gencode_translations, gencode_metadata = loader.load_gencode()

    # (2) PROCESS 
    # (a) include transcripts with a median TPM > 5 in ANY tissue
    hpa_df = tpm_filter(hpa_df, hpa_metadata['tissues'], hpa_metadata['tpm_columns'], plotting=False)
    
    # (b) include transcripts which work as advertised (translating them gives valid proteins)
    df = quality_control(gencode_transcripts, gencode_translations, gencode_metadata, hpa_df)
    logging.info(f"Found {len(df):,} matched transcripts")

    # (c) filter for length
    df = df[(df['DNA_seq'].str.len() >= 200) & (df['DNA_seq'].str.len() <= 2000)]
    logging.info(f"Found {len(df):,} transcripts matching length criteria")

    # (d) calculate CAI and MFE, include only transcripts that fall within the criteria 
    threads = min(multiprocessing.cpu_count(), 64)
    dataframe_to_fasta(df, 'DNA_seq', 'data/filtered_transcripts.fa')

    cai_df = run_cai(fasta_path="data/filtered_transcripts.fa")
    # cai_df = run_cai_multicore(fasta_path="data/filtered_transcripts.fa", num_processes=threads) BUG: doesn't save values properly!
    mfe_df = parallel_mfe(cai_df, threads=threads)
    mfe_df.to_csv("data/mfe_cai.tsv", sep="\t", index=False)

    final_df = mfe_df[(mfe_df['CAI'] >= 0.7) & (mfe_df['MFE'] <= -200)]
    final_df.to_csv("data/train_data.tsv", sep="\t", index=False)

    # (3) SAVE
    
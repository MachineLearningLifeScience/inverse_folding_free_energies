import math
import glob 
import os
import pathlib
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd

## Verify that pytorch-geometric is correctly installed
import torch_geometric
import torch_sparse
from torch_geometric.nn import MessagePassing

import esm
from esm.inverse_folding.util import CoordBatchConverter
from esm.inverse_folding.util import load_structure, extract_coords_from_structure
from biotite.structure.residues import get_residues

import warnings
warnings.filterwarnings("ignore", message="Regression weights not found, predicting contacts will not produce correct results.")


def get_structure_data(structure_filename, chain_id, resid_offset=0, verbose=False):
    """Extract structure data in ESM format"""

    structure = load_structure(structure_filename, chain_id)
    pdb_coords, pdb_seq = extract_coords_from_structure(structure)    

    residue_mapper = {}
    pdb_resid_list = get_residues(structure)[0]
    assert(len(pdb_resid_list) == len(pdb_seq))
    for idx, (res_id, aa) in enumerate(zip(pdb_resid_list, pdb_seq)):
        residue_mapper[res_id + resid_offset] = idx    

    if verbose:
        pdb_full_seq = ["."]*(max(pdb_resid_list)+resid_offset+1)
        for idx, (res_id, aa) in enumerate(zip(pdb_resid_list, pdb_seq)):
            pdb_full_seq[res_id + resid_offset] = aa
        pdb_full_seq = "".join(pdb_full_seq)
        print(pdb_full_seq)

    return pdb_coords, pdb_seq, residue_mapper


# Set evaluate_only_at_position=index if you only wish to evaluate the conditional p(s_i | s_{\i}) at the site that is changed.
# Since we only look at single mutations, this is sufficient for our purposes, since p(s_{\i}) will cancel in the ration between WT and MT
# def evaluate_likelihood(model, batch, alphabet, device, evaluate_only_at_position=None):
def calc_likelihood(model, batch, alphabet, device):
    """Call ESM2 likelihood evaluation"""

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(batch)

    # Check that all strings in batch have the same length
    assert len({len(item[1]) for item in batch}) == 1
    seq_len = len(batch[0][1])
    
    log_probs = []
    for i in range(1, batch_tokens.shape[1]-1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[:, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.to(device))["logits"], dim=-1)
        log_probs.append(token_probs[:, i][torch.arange(batch_tokens.shape[0]), batch_tokens[:, i]])
    log_probs = torch.vstack(log_probs).T
    return log_probs.sum(axis=1).cpu().numpy()




def calculate_likelihoods(model, alphabet, ddg_data, sequence, pdb_id, index_offset=0, output_dir="output", output_file_prefix="ddgs_", batch_size=8, device="cpu", use_cache=False, verbose=False):
    """Evaluate likelihoods using the ESM2 model"""

    df = (ddg_data[ddg_data['pdbid']==pdb_id]).copy()

    df['ESM2'] = np.nan

    # chain_id = structure_chain_id
    # if chain_id is None:
    #     chain_id = df['chainid'].iloc[0]
    ddg_resid_list = df['variant'].str[1:-1].astype(int).tolist()
    ddg_seq = df['variant'].str[0].tolist()

    residue_mapper = {}
    for idx, (res_id, aa) in enumerate(zip(ddg_resid_list, ddg_seq)):
        residue_mapper[res_id] = res_id + index_offset    

    if verbose:
        # ddg_full_seq = ["."]*(max(ddg_resid_list)+1)
        # for res_id, aa in zip(ddg_resid_list, ddg_seq):
        #     ddg_full_seq[res_id] = aa        
        # ddg_full_seq = "".join(ddg_full_seq)
        # print(ddg_full_seq)

        seq_obs = ["."]*(len(sequence))
        for res_id, aa in zip(ddg_resid_list, ddg_seq):
            idx_seq = residue_mapper[res_id] - 1
            try:
                seq_obs[idx_seq] = aa
            except IndexError:
                print("Warning: index {} is out of range of sequence of length {}".format(idx_seq, len(seq_obs)))
        print(sequence)
        print("".join(seq_obs))

    # structure_file_id = pathlib.PurePath(structure_filename).parts[-1]

    output_filename = f'{output_dir}/{output_file_prefix}{pdb_id}.csv'

    if use_cache:
        if os.path.exists(output_filename):
            return pd.read_csv(output_filename)
        else:
            output_filename = output_filename.replace(pdb_id.lower(), pdb_id.upper())
            if os.path.exists(output_filename):
                return pd.read_csv(output_filename)
            else:
                # If output_filename was not found, check lower case name
                output_filename = output_filename.replace(pdb_id.upper(), pdb_id.lower())
                if os.path.exists(output_filename):
                    return pd.read_csv(output_filename)
            

        assert False, f"CACHE-FILE NOT FOUND: {output_filename}"

    # _, pdb_seq, residue_mapper = get_structure_data(structure_filename, chain_id, resid_offset=structure_index_offset, verbose=verbose)

    # Evaluate WT likelihood
    batch = [(None, sequence)]
    wt_likelihood = calc_likelihood(model, batch, alphabet, device)[0]

    for start_idx in range(0, len(df), batch_size):
        df_batch = df[start_idx:start_idx+batch_size]

        batch = []
        indices = []
        for idx, row in df_batch.iterrows():

            variant = row['variant']
            if pd.isna(variant):
                continue
            v_from = variant[0]
            v_to = variant[-1]
            position = int(variant[1:-1])

            try:
                idx_seq = residue_mapper[position] - 1
            except KeyError:
                print(f"WARNING ({pdb_id}): skipping position {position} because no information was found in provided PDB")
                continue

            try:
                assert v_from == sequence[idx_seq], f"{v_from},{sequence[idx_seq]}"
                seq = sequence[:idx_seq] + v_to + sequence[idx_seq+1:]

            except AssertionError:
                print(f"WARNING ({pdb_id}): skipping position {position} because of wildtype mismatch: {v_from} vs {sequence[idx_seq]}")
                continue
            except IndexError:
                print(f"WARNING ({pdb_id}): skipping position {position} because no information was found in provided PDB")
                continue

            batch.append((None, seq))
            
            indices.append(idx)

            # print("SUCCESS")

        if len(batch)==0:
            continue
        
        # Check whether batch contains unequal batch sizes, and if so, split
        if len(set([len(item[1]) for item in batch])) > 1:
            print("Not of equal length")
            likelihoods = np.zeros(len(batch))
            subbatches = {}
            for i, item in enumerate(batch):
                seq_len = len(item[2])
                if seq_len not in subbatches:
                    subbatches[seq_len] = {'indices':[], 'entries':[]}
                subbatches[seq_len]['indices'].append(i)
                subbatches[seq_len]['entries'].append(item)
            for seq_len in subbatches:
                subbatch_indices = subbatches[seq_len]['indices']
                subbatch = subbatches[seq_len]['entries']
                sublikelihoods = calc_likelihood(model, subbatch, alphabet, device)
                for j, likelihood in enumerate(sublikelihoods):
                    likelihoods[subbatch_indices[j]] = likelihood
        else:
            likelihoods = calc_likelihood(model, batch, alphabet, device)

        print(likelihoods)
        df.loc[indices, 'ESM2'] = likelihoods

        # Add wildtype score if structure file was given for entire dataset
        df.loc[len(df)] = {'pdbid':pdb_id, 'chainid':df['chainid'].iloc[0], 'ESM2': wt_likelihood}

        df.to_csv(output_filename, index=False)

    return df

if __name__ == "__main__":

    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-input-file', help="CSV file containing variants")
    parser.add_argument('--device', default="cuda", help="cpu|mps|cuda")
    parser.add_argument('--batch-size', default="16", help="Size of batch used for evaluation", type=int)
    parser.add_argument('--id', help="PDB ID in CSV file")
    parser.add_argument('--output-dir', default="output", help="Output directory")
    parser.add_argument('--output-file-prefix', help="Output file prefix", default="results_")
    parser.add_argument('--index-offset', help="Offset to add to PDB resid", default=0, type=int)
    parser.add_argument('--sequence', help="amino acid sequence")
    parser.add_argument('--verbose', help="Whether to print debug information", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    # Create output dir if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    

    print(args.id)

    # Create model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(args.device)
    model.eval().requires_grad_(False)

    ddg_data = pd.read_csv(args.data_input_file)
    residue_index_map = None

    calculate_likelihoods(model, alphabet, ddg_data,
                          sequence=args.sequence, pdb_id=args.id, index_offset=args.index_offset,                          
                          output_dir=args.output_dir, output_file_prefix=args.output_file_prefix, batch_size = args.batch_size, 
                          device=args.device, verbose=args.verbose)





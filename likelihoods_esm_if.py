import os
import pathlib
from Bio import SeqIO
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import esm
from esm.inverse_folding.util import CoordBatchConverter
from esm.inverse_folding.util import load_structure, extract_coords_from_structure
from biotite.structure.residues import get_residues


def get_esm_structure_data(structure_filename, chain_id, resid_offset=0, verbose=False):
    """Extract structure data in the format ESM-IF expects"""

    structure = load_structure(structure_filename, chain_id)
    pdb_coords, pdb_seq = extract_coords_from_structure(structure)    

    residue_mapper = {}

    pdb_resid_list = get_residues(structure)[0]
    assert(len(pdb_resid_list) == len(pdb_seq))
    for idx, (res_id, aa) in enumerate(zip(pdb_resid_list, pdb_seq)):
        # pdb_full_seq[res_id + resid_offset] = aa
        residue_mapper[res_id + resid_offset] = idx    

    # Optionally print out sequence embedded in full length gaps
    if verbose:
        pdb_full_seq = ["."]*(max(pdb_resid_list)+resid_offset+1)
        for idx, (res_id, aa) in enumerate(zip(pdb_resid_list, pdb_seq)):
            pdb_full_seq[res_id + resid_offset] = aa
        pdb_full_seq = "".join(pdb_full_seq)
        print(pdb_full_seq)

    return pdb_coords, pdb_seq, residue_mapper


def calc_likelihood(model, batch, alphabet, device):
    """Call ESM-IF likelihood evaluation"""

    batch_converter = CoordBatchConverter(alphabet)
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)

    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:].to(device)
    target_padding_mask = (target == alphabet.padding_idx)
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
    loss = F.cross_entropy(logits, target, reduction='none')
    loss = loss.cpu().detach().numpy()
    target_padding_mask = target_padding_mask.cpu().numpy()
    ll_fullseq = -np.sum(loss * ~target_padding_mask, axis=1)
    return ll_fullseq


def calc_likelihoods(model, alphabet, ddg_data, structure_filename, pdb_id, structure_filename_pattern=None, structure_chain_id='A', structure_index_offset=0, output_dir="output", output_file_prefix="ddgs_", batch_size=8, device="cpu", pdb_is_mutant=False, use_cache=False, evaluate_pdb_seq=False, verbose=False, evaluate_fragment=False, fragment_length=3):
    """Evaluate likelihoods using the ESM-IF model"""

    # Copy input dataframe and add output column
    df = (ddg_data[ddg_data['pdbid']==pdb_id]).copy()
    df['ESM-IF'] = np.nan

    # Use provided chain_id if given, otherwise default to chain id provided in dataframe
    chain_id = structure_chain_id
    if chain_id is None:
        chain_id = df['chainid'].iloc[0]
    # Extract residue numbers and seq indices
    ddg_resid_list = df['variant'].str[1:-1].astype(int).tolist()
    ddg_seq = df['variant'].str[0].tolist()

    # Optionally print out partial sequences embedded surrounded by full length gaps
    if verbose:
        ddg_full_seq = ["."]*(max(ddg_resid_list)+1)
        for res_id, aa in zip(ddg_resid_list, ddg_seq):
            ddg_full_seq[res_id] = aa        
        ddg_full_seq = "".join(ddg_full_seq)
        print(ddg_full_seq)

    # If a structure filename is given use that. Otherwise, use the structure_filename_pattern to specify multiple PDB files at once
    if structure_filename:
        structure_file_id = pathlib.PurePath(structure_filename).parts[-1]
    else: 
        structure_file_id = pathlib.PurePath(structure_filename_pattern).parts[-1]

    # Set output filename
    output_filename = f'{output_dir}/{output_file_prefix}{structure_file_id}.csv'

    #  If use_cache, check for precalculated likelihoods
    if use_cache:
        if os.path.exists(output_filename):
            return pd.read_csv(output_filename)
        else:
            # If output_filename was not found, check upper case name
            output_filename = output_filename.replace(pdb_id.lower(), pdb_id.upper())
            if os.path.exists(output_filename):
                return pd.read_csv(output_filename)

        assert False, f"CACHE-FILE NOT FOUND: {output_filename}"

    # If structure_filename is provided, use same structure file for entire dataset
    if structure_filename:
        assert structure_filename_pattern is None

        pdb_coords, pdb_seq, residue_mapper = get_esm_structure_data(structure_filename, chain_id, resid_offset=structure_index_offset, verbose=verbose)

        if not evaluate_fragment:
            # Evaluate WT likelihood
            batch = [(pdb_coords, None, pdb_seq)]
            wt_likelihood = calc_likelihood(model, batch, alphabet, device)[0]

    # Iterate over entries in input dataframe
    for start_idx in range(0, len(df), batch_size):

        # Construct batch
        df_batch = df[start_idx:start_idx+batch_size]
        batch = []
        indices = []
        for idx, row in df_batch.iterrows():

            # Extract variant and position
            variant = row['variant']
            if pd.isna(variant):  # Skip entry if not defined
                continue
            v_from = variant[0]
            v_to = variant[-1]
            position = int(variant[1:-1])

            # If structure_filename was not provided, use structure_filename_pattern to dynamically construct PDB filename based on variant
            if structure_filename_pattern:
                assert structure_filename is None
                pdb_filename = structure_filename_pattern.format(variant)
                if not os.path.exists(pdb_filename):
                    print(f"WARNING ({pdb_filename}): skipping variant {variant} because no PDB was found for pattern")
                    continue
                # Retrieve structural data
                pdb_coords, pdb_seq, residue_mapper = get_esm_structure_data(pdb_filename, chain_id, resid_offset=position, verbose=verbose)

            # Index mapper translates between the residue position provided in the experimental dataframe, and the index in the PDB file
            try:
                idx_seq = residue_mapper[position]
            except KeyError:
                print(f"WARNING ({structure_file_id}): skipping position {position} because no information was found in provided PDB")
                continue

            try:
                # Normally, the provided PDB file is always the WT structure. This can be overridden here
                if pdb_is_mutant:
                    assert v_to == pdb_seq[idx_seq], f"{v_to},{pdb_seq[idx_seq]}"
                else:
                    assert v_from == pdb_seq[idx_seq], f"{v_from},{pdb_seq[idx_seq]}"

                # If evaluate_pdb_seq is true, we evaluate the seq given in the PDB structure, rather than the one in the experimental data
                if evaluate_pdb_seq:
                    seq = pdb_seq
                else:
                    seq = pdb_seq[:idx_seq] + v_to + pdb_seq[idx_seq+1:]

            except AssertionError:
                print(f"WARNING ({structure_file_id}): skipping position {position} because of wildtype mismatch: {v_from} vs {pdb_seq[idx_seq]}")
                continue
            except IndexError:
                print(f"WARNING ({structure_file_id}): skipping position {position} because no information was found in provided PDB")
                continue

            # If evaluate_fragment, we only evaluate a subsequence
            if evaluate_fragment:
                batch.append((pdb_coords[max(0, idx_seq-fragment_length//2):min(len(pdb_coords), idx_seq+fragment_length//2+1)], 
                              None, 
                              seq[max(0, idx_seq-fragment_length//2):min(len(seq), idx_seq+fragment_length//2+1)]))
            else:
                batch.append((pdb_coords, None, seq))
            
            indices.append(idx)

        # Do nothing if batch is empty
        if len(batch)==0:
            continue
        
        # Check whether batch contains unequal batch sizes, and if so, split
        if len(set([len(item[2]) for item in batch])) > 1:
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

        df.loc[indices, 'ESM-IF'] = likelihoods

        # Add wildtype score if structure file was given for entire dataset
        if structure_filename and not evaluate_fragment:
            df.loc[len(df)] = {'pdbid':pdb_id, 'chainid':df['chainid'].iloc[0], 'ESM-IF': wt_likelihood}

        # Dump to CSV
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
    parser.add_argument('--pdb', help="PDB-file", default=None)
    parser.add_argument('--pdb-pattern', help="PDB pattern", default=None)
    parser.add_argument('--pdb-is-mutant', help="Whether PDB contains mutant", default=0, type=bool)
    parser.add_argument('--pdb-chain-id', help="Chain ID in PDB file. If not specifies, use the one specified in the ddg file", default=None)
    parser.add_argument('--pdb-index-offset', help="Offset to add to PDB resid", default=0, type=int)
    parser.add_argument('--evaluate-pdb-seq', help="Evaluate sequence in PDB file (rather than mutant)", action=argparse.BooleanOptionalAction)
    parser.add_argument('--evaluate-fragment', help="Whether to evaluate only a fragment rather than the whole PDB", action=argparse.BooleanOptionalAction)
    parser.add_argument('--fragment-length', help="Length of fragments to consider", default=3, type=int)
    parser.add_argument('--verbose', help="Whether to print debug information", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    # Create output dir if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    

    print(args.pdb, args.pdb_pattern)

    # Create model
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.to(args.device)
    model = model.eval().requires_grad_(False)

    ddg_data = pd.read_csv(args.data_input_file)
    residue_index_map = None


    calc_likelihoods(model, alphabet, ddg_data, 
                     structure_filename=args.pdb, structure_filename_pattern=args.pdb_pattern, pdb_id=args.id,
                     structure_chain_id=args.pdb_chain_id, structure_index_offset=args.pdb_index_offset,
                     output_dir=args.output_dir, output_file_prefix=args.output_file_prefix, batch_size = args.batch_size, 
                     device=args.device, pdb_is_mutant=args.pdb_is_mutant, evaluate_pdb_seq=args.evaluate_pdb_seq, 
                     evaluate_fragment=args.evaluate_fragment, fragment_length=args.fragment_length,
                     verbose=args.verbose)

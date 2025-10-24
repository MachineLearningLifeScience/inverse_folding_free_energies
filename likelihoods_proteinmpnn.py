# Slightly modified version of the protein_mpnn_run.py script provided in https://github.com/dauparas/ProteinMPNN/

import argparse
import os.path

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

import copy

import pathlib
import numpy as np
import pandas as pd

from esm.inverse_folding.util import load_structure
from biotite.structure.residues import get_residues

from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

def get_structure_data(structure_filename, chain_id, resid_offset=0, verbose=False):
    """Extract structure data from PDB file"""

    pdb_dict_list = parse_PDB(structure_filename, ca_only=False)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=200000)

    # Our script only handles a single protein at a time
    assert len(dataset_valid) == 1
    
    all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
    pdb_path_chains = chain_id
    if pdb_path_chains:
        designed_chain_list = [str(item) for item in pdb_path_chains.split()]
    else:
        designed_chain_list = all_chain_list
    fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

    protein = dataset_valid[0]
    pdb_seq = protein['seq']

    # ProteinMPNN parser does not contain seqid information, using the one from ESM to construct index mapper
    structure = load_structure(structure_filename, chain_id)

    residue_mapper = {}

    pdb_resid_list = get_residues(structure)[0]
    assert(len(pdb_resid_list) == len(pdb_seq))
    for idx, (res_id, aa) in enumerate(zip(pdb_resid_list, pdb_seq)):
        # pdb_full_seq[res_id + resid_offset] = aa
        residue_mapper[res_id + resid_offset] = idx    

    if verbose:
        pdb_full_seq = ["."]*(max(pdb_resid_list)+resid_offset+1)
        for idx, (res_id, aa) in enumerate(zip(pdb_resid_list, pdb_seq)):
            pdb_full_seq[res_id + resid_offset] = aa
        pdb_full_seq = "".join(pdb_full_seq)
        print(pdb_full_seq)
    
    return protein, pdb_seq, chain_id_dict, residue_mapper


def calc_likelihood(model, batch, chain_id_dict, device):
    """Call ProteinMPNN likelihood evaluation"""

    fixed_positions_dict = None
    omit_AA_dict = None
    tied_positions_dict = None
    pssm_dict = None
    bias_by_res_dict = None

    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(alphabet, range(21)))

    protein_batch = [item[0] for item in batch]
    seq_batch = [item[-1] for item in batch]
    
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(protein_batch, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=False)
    
    randn_1 = torch.randn(chain_M.shape, device=X.device)

    for i, seq in enumerate(seq_batch):
        S_input = torch.tensor([alphabet_dict[AA] for AA in seq], device=device)[None,:]#.repeat(X.shape[0], 1)
        S[i:i+1,:len(seq)] = S_input
    
    log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
    mask_for_loss = mask*chain_M*chain_M_pos
    ll_fullseq = -_scores(S, log_probs, mask_for_loss)  # _scores returns negative log likslihoods
    ll_fullseq *= torch.sum(mask_for_loss, dim=-1)      # _scores normalizes by length. We don't want that
    return ll_fullseq.cpu().detach().numpy()


    
def calc_likelihoods(model, ddg_data, structure_filename, pdb_id, structure_filename_pattern=None, structure_chain_id='A', structure_index_offset=0, output_dir="output", output_file_prefix="ddgs_", batch_size=8, device="cpu", pdb_is_mutant=False, use_cache=False, evaluate_pdb_seq=False, verbose=False, evaluate_fragment=False):
    """Evaluate likelihoods using Protein MPNN"""

    df = (ddg_data[ddg_data['pdbid']==pdb_id]).copy()
    
    df['MPNN'] = np.nan

    chain_id = structure_chain_id
    if chain_id is None:
        chain_id = df['chainid'].iloc[0]
    ddg_resid_list = df['variant'].str[1:-1].astype(int).tolist()
    ddg_seq = df['variant'].str[0].tolist()

    if verbose:
        ddg_full_seq = ["."]*(max(ddg_resid_list)+1)
        for res_id, aa in zip(ddg_resid_list, ddg_seq):
            ddg_full_seq[res_id] = aa        
        ddg_full_seq = "".join(ddg_full_seq)
        print(ddg_full_seq)

    if structure_filename:
        structure_file_id = pathlib.PurePath(structure_filename).parts[-1]
    else: 
        structure_file_id = pathlib.PurePath(structure_filename_pattern).parts[-1]

    output_filename = f'{output_dir}/{output_file_prefix}{structure_file_id}.csv'

    if use_cache:
        if os.path.exists(output_filename):
            return pd.read_csv(output_filename)
        else:
            output_filename = output_filename.replace(pdb_id.lower(), pdb_id.upper())
            if os.path.exists(output_filename):
                return pd.read_csv(output_filename)

        assert False, f"CACHE-FILE NOT FOUND: {output_filename}"

    if structure_filename:
        assert structure_filename_pattern is None

        protein, pdb_seq, chain_id_dict, residue_mapper = get_structure_data(structure_filename, chain_id, resid_offset=structure_index_offset, verbose=verbose)
        
        if not evaluate_fragment:
            batch = [(protein, None, protein['seq'])]
            wt_likelihood = calc_likelihood(model, batch, chain_id_dict, device)[0]
            
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

            if structure_filename_pattern:
                assert structure_filename is None
                pdb_filename = structure_filename_pattern.format(variant)
                if not os.path.exists(pdb_filename):
                    print(f"WARNING ({pdb_filename}): skipping variant {variant} because no PDB was found for pattern")
                    continue
                protein, pdb_seq, chain_id_dict, residue_mapper = get_structure_data(pdb_filename, chain_id, resid_offset=position, verbose=verbose)

            try:
                idx_seq = residue_mapper[position]
            except KeyError:
                print(f"WARNING ({structure_file_id}): skipping position {position} because no information was found in provided PDB")
                continue

            try:
                if pdb_is_mutant:
                    assert v_to == pdb_seq[idx_seq], f"{v_to},{pdb_seq[idx_seq]}"
                else:
                    assert v_from == pdb_seq[idx_seq], f"{v_from},{pdb_seq[idx_seq]}"

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
            
            if evaluate_fragment:
                fragment_length = 3
                protein_fragment = copy.deepcopy(protein)
                protein_fragment['seq_chain_A'] = protein['seq_chain_A'][max(0, idx_seq-fragment_length//2):min(len(seq), idx_seq+fragment_length//2+1)] 
                protein_fragment['seq'] = protein['seq'][max(0, idx_seq-fragment_length//2):min(len(seq), idx_seq+fragment_length//2+1)]
                for atomtype in protein['coords_chain_A']:
                    protein_fragment['coords_chain_A'][atomtype] = protein['coords_chain_A'][atomtype][max(0, idx_seq-fragment_length//2):min(len(seq), idx_seq+fragment_length//2+1)]
                batch.append((protein_fragment, 
                              None, 
                              seq[max(0, idx_seq-fragment_length//2):min(len(seq), idx_seq+fragment_length//2+1)]))
            else:
                batch.append((protein, None, seq))
            
            indices.append(idx)

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
                sublikelihoods = calc_likelihood(model, subbatch, chain_id_dict, device)
                for j, likelihood in enumerate(sublikelihoods):
                    likelihoods[subbatch_indices[j]] = likelihood
        else:
            likelihoods = calc_likelihood(model, batch, chain_id_dict, device)

        print(likelihoods)

        df.loc[indices, 'MPNN'] = likelihoods

        # Add wildtype score if structure file was given for entire dataset
        if structure_filename and not evaluate_fragment:
            df.loc[len(df)] = {'pdbid':pdb_id, 'chainid':df['chainid'].iloc[0], 'MPNN': wt_likelihood}

        df.to_csv(output_filename, index=False)

    return df
        
    
if __name__ == "__main__":

    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-input-file', help="CSV file containing variants")
    # parser.add_argument('--data-reference-file', help="CSV file containing reference information (ProteinGym)")
    parser.add_argument('--device', default="cuda", help="cpu|mps|cuda")
    parser.add_argument('--model-path-dir', help="path to model parameter checkpoint directory")
    parser.add_argument('--model-name', type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
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
    parser.add_argument('--verbose', help="Whether to print debug information", action=argparse.BooleanOptionalAction)
    # parser.add_argument('--residue-index-map', help="txt file containing mapping between equivalent residue indices.")

    args = parser.parse_args()

    # Create output dir if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    

    print(args.pdb, args.pdb_pattern)

    # Create model
    checkpoint_path = args.model_path_dir + f'/{args.model_name}.pt'
    device = args.device
    checkpoint = torch.load(checkpoint_path, map_location=device)

    hidden_dim = 128
    num_layers = 3
    backbone_noise = 0.0
    
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])

    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()    

    ddg_data = pd.read_csv(args.data_input_file)
    # reference_data = pd.read_csv(args.data_reference_file)

    residue_index_map = None    
    
    calc_likelihoods(model, ddg_data, #reference_data, 
                          structure_filename=args.pdb, structure_filename_pattern=args.pdb_pattern, pdb_id=args.id,
                          structure_chain_id=args.pdb_chain_id, structure_index_offset=args.pdb_index_offset,
                          output_dir=args.output_dir, output_file_prefix=args.output_file_prefix, batch_size = args.batch_size, 
                          device=args.device, pdb_is_mutant=args.pdb_is_mutant, evaluate_pdb_seq=args.evaluate_pdb_seq, 
                          evaluate_fragment=args.evaluate_fragment,
                          verbose=args.verbose)



    
#     model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
#     model.to(args.device)
#     model = model.eval().requires_grad_(False)
    


# model.to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
    

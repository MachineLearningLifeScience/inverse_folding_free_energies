import mdtraj
import numpy as np
import sys
from Bio import Align
from itertools import combinations

# Code adapted from https://github.com/microsoft/bioemu-benchmarks to ensure identical definition of FNC

"""Taken from https://www.cup.uni-muenchen.de/ch/compchem/tink/as.html"""

RESTYPE_1TO3: dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "U": "SEC",  # Selenocysteine
    "O": "PYL",  # Pyrrolysine
    "X": "UNK",  # Unknown
    "B": "NLE",  # Norleucine, Found in DEShaw WW_DOMAIN, isomer of Leucine found in some bacterial strains.
    "Z": "GLX",  # Glutamine or glutamic acid. Used for when we can't distinguish between GLU and GLN.
}

RESTYPE_3TO1: dict[str, str] = {v: k for k, v in RESTYPE_1TO3.items()}


def get_aa1code_from_aa3code(aa3code: str) -> str:
    if aa3code in RESTYPE_3TO1:
        return RESTYPE_3TO1[aa3code]
    else:
        return "X"


def range_to_all_matching_resids(matching_ranges: np.ndarray) -> list[tuple[int, int]]:
    """
    Converts a list of aligned residue index ranges between two sequences into a list of tuples of explicit aligned
    residue indices.

    Args:
        matching_ranges : List of aligned residue ranges, as present in the `Bio.Alignment.aligned` attribute
                            (e.g., (e.g. [[[0, 154], [0, 154]], [[156, 168], [156, 168]], ...]))

    Returns:
        A list of tuples with explicitly aligned indices between two sequences(e.g. [(0, 0), (1, 1), ...])
    """
    ranges_i, ranges_j = matching_ranges

    matching_resids: list[tuple[int, int]] = []
    for range_row_i, range_row_j in zip(ranges_i, ranges_j):
        start_i, end_i = range_row_i
        start_j, end_j = range_row_j

        start_to_end_i = range(start_i, end_i)
        start_to_end_j = range(start_j, end_j)
        matching_resids.extend(list(zip(start_to_end_i, start_to_end_j)))
    return matching_resids


def seq_pairwise_align_trajs(
    traj_i: mdtraj.Trajectory, traj_j: mdtraj.Trajectory
) -> list[tuple[int, int]]:
    """Gets matching pairs of residues between `traj_i` and `traj_j` via
    global pairwise sequence alignment.

    Args:
        traj_i: First trajectory
        traj_j: Second trajectory

    Returns:
        List of tuples with matching residue indices between `traj_i` and `traj_j`.
    """

    def _get_seq_resid_traj(traj: mdtraj.Trajectory) -> tuple[list[int], str]:
        residues = list(traj.topology.residues)
        resids = [r.resSeq for r in residues]
        resnames = [r.name for r in residues]
        return resids, "".join([get_aa1code_from_aa3code(r) for r in resnames])

    resid_i, seq_i = _get_seq_resid_traj(traj_i)
    resid_j, seq_j = _get_seq_resid_traj(traj_j)
    aligner = Align.PairwiseAligner(mode="global", open_gap_score=-0.5)
    alignment = aligner.align(seq_i, seq_j)[
        0
    ]  # Get first alignment (max score) if more than one available
    matching_ranges = alignment.aligned
    matching_resid_zero_idx = range_to_all_matching_resids(matching_ranges)
    return [(resid_i[i], resid_j[j]) for i, j in matching_resid_zero_idx]


def compute_contacts(
    traj_i: mdtraj.Trajectory,
    traj_j: mdtraj.Trajectory,
    matching_resids: list[tuple[int, int]] | None = None,
    reference_resid_pairs: list[tuple[int, int]] | None = None,
    threshold: float = 8.0,
    exclude_n_neighbours: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes contact maps for two trajectories over a set of common residues specified by `matching_resids`.

    Args:
        traj_i: First trajectory. Note: this trajectory is used as reference to determine which residues
                are sequence neighbours.
        traj_j: Second trajectory.
        matching_resids: Matching resSeq indices between the trajectories
        reference_resid_pairs: If passed, contacts will only be computed between pairs of specified
                            residue resSeqs.
        threshold: Distance threshold (in angstroms) to define contacts.
        exclude_n_neighbours: If set to > 0, contact maps will be excluded between residues whose residue index
                            differences is fewer or equal to this number.

    Returns
        * `contacts_i`, `contacts_j`: Arrays with contact maps
        * A list of pairs of residue indices corresponding to `traj_i`
          over which the contact map was computed
        * A list of pairs of zero-based residue indices corresponding to `traj_i`
          over which the contact map was computed
    """
    if matching_resids is None:
        matching_resids = seq_pairwise_align_trajs(traj_i, traj_j)

    # Get zero-indexed residue indices from matching_resids
    def _get_zero_index_resid_map(traj: mdtraj.Trajectory, resids: list[int]) -> dict[int, int]:
        resids_set = set(resids)
        resids_to_zeroidx_resid: dict[int, int] = {}
        for residue in traj.topology.residues:
            if residue.resSeq in resids_set:
                resids_to_zeroidx_resid[residue.resSeq] = residue.index
        return resids_to_zeroidx_resid

    resids_i, resids_j = [m[0] for m in matching_resids], [m[1] for m in matching_resids]
    resid_i_to_j = {k: v for k, v in matching_resids}

    # Exclude neighbouring residues using `traj_i` as a reference.
    valid_resid_combs_i = [
        (r, l) for r, l in list(combinations(resids_i, 2)) if abs(r - l) >= exclude_n_neighbours
    ]

    if reference_resid_pairs is not None:
        reference_resid_pairs_set = set(reference_resid_pairs)
        valid_resid_combs_i = [
            (r, l)
            for r, l in valid_resid_combs_i
            if (r, l) in reference_resid_pairs_set or (l, r) in reference_resid_pairs_set
        ]

    valid_resid_combs_i = np.array(valid_resid_combs_i)
    valid_resid_combs_j = np.array(
        [(resid_i_to_j[r], resid_i_to_j[l]) for r, l in valid_resid_combs_i]
    )

    # Translate resids to zero-based resids to use `compute_contacts`
    resid_to_zeroidx_resid_i = _get_zero_index_resid_map(traj_i, resids_i)
    resid_to_zeroidx_resid_j = _get_zero_index_resid_map(traj_j, resids_j)

    valid_zeroidx_resid_combs_i = np.array(
        [
            (resid_to_zeroidx_resid_i[resid_r], resid_to_zeroidx_resid_i[resid_l])
            for resid_r, resid_l in valid_resid_combs_i
        ]
    )
    valid_zeroidx_resid_combs_j = np.array(
        [
            (resid_to_zeroidx_resid_j[resid_r], resid_to_zeroidx_resid_j[resid_l])
            for resid_r, resid_l in valid_resid_combs_j
        ]
    )

    distances_i, _ = mdtraj.compute_contacts(
        traj_i, scheme="ca", contacts=valid_zeroidx_resid_combs_i
    )
    distances_j, _ = mdtraj.compute_contacts(
        traj_j, scheme="ca", contacts=valid_zeroidx_resid_combs_j
    )

    # Convert to angstrom
    distances_i *= 10.0
    distances_j *= 10.0

    contacts_i, contacts_j = (
        (distances_i < threshold).astype(int),
        (distances_j < threshold).astype(int),
    )
    return contacts_i, contacts_j, valid_resid_combs_i, valid_zeroidx_resid_combs_i


def fraction_native_contacts(
    traj_i: mdtraj.Trajectory,
    traj_j: mdtraj.Trajectory,
    matching_resids: list[tuple[int, int]] | None = None,
    reference_resid_pairs: list[tuple[int, int]] | None = None,
    threshold: float = 8.0,
    exclude_n_neighbours: int = 0,
) -> np.ndarray:
    """Computes fraction of native contacts between a reference trajectory `traj_i` and another one `traj_j`,
    over a region of matching residues `matching_resids`. Only positive contacts in `traj_i` are used for
    comparison.

    Args:
        traj_i: Reference trajectory
        traj_j: Trajectory to compare against
        matching_resids: Matching resSeq indices between the trajectories.
        reference_resid_pairs: If passed, contacts will only be computed between pairs of specified
                            residue resSeqs
        threshold: Distance threshold (in Angstrom) to consider two residues as contacting.
        exclude_n_neighbours: If set to > 0, contact maps will be excluded between residues whose residue index
                            differences is fewer or equal to this number.
    """
    contacts_i, contacts_j, _, _ = compute_contacts(
        traj_i,
        traj_j,
        matching_resids=matching_resids,
        reference_resid_pairs=reference_resid_pairs,
        threshold=threshold,
        exclude_n_neighbours=exclude_n_neighbours,
    )
    native_contact_indices = np.where(contacts_i[0, :] == 1)[0]
    return np.mean(
        contacts_i[:, native_contact_indices] == contacts_j[:, native_contact_indices], axis=1
    )


if __name__ == "__main__":

    reference_traj = sys.argv[1]
    target_traj = sys.argv[2:]

    for filename in target_traj:
        print(fraction_native_contacts(mdtraj.load(reference_traj), mdtraj.load(filename)))

"""
Decodes the chromosome's optional weighting-scheme gene, and works out
where the penalty-matrix genes end and that gene begins.

The paper's own chromosome (Section II-B/II-C) searches over the penalty
matrix AND a choice of weighting scheme (INS/ISNS/ENS) together — "the
number of chromosomes that we use to encode the search space depends on
the number of networks after the input dispatcher... we need (num. of
DNNs)^2 chromosomes to encode the P matrix and an extra one for the
weighting scheme." This repo's shared dispatcher/ code has historically
hardcoded INS only (a settled decision for ImageNette and CIFAR-10, see
CLAUDE.md — not revisited for those two tracks). Whether a given track
searches over the scheme too is inferred structurally from its
config.N_GENES, not a separate boolean flag:

  - config.N_GENES == NUM_CLASSES**2 - NUM_CLASSES
        -> no scheme gene. Every individual uses INS, exactly as before.
  - config.N_GENES == NUM_CLASSES**2 - NUM_CLASSES + 1
        -> the chromosome's last gene selects the scheme. Needs
           config.SCHEME_GENE_LOWER_BOUND/SCHEME_GENE_UPPER_BOUND defined
           (the search range pymoo samples that one gene from — a
           different range than the penalty genes' [0, 100]-ish bounds).

This lets a track opt into the extra gene just by sizing N_GENES
correctly, without every shared function needing an explicit "does this
track search schemes" parameter threaded through it.
"""

import numpy as np

WEIGHTING_SCHEMES = ["INS", "ISNS", "ENS"]


def num_penalty_genes(config):
    """Off-diagonal entries of the NUM_CLASSES x NUM_CLASSES penalty
    matrix — always this many regardless of whether a scheme gene is
    also present."""
    return config.NUM_CLASSES ** 2 - config.NUM_CLASSES


def searches_weighting_scheme(config):
    """True if this track's chromosome has the extra scheme-selector gene
    (config.N_GENES is one more than the penalty-only count), False if it
    only ever uses INS (config.N_GENES matches the penalty-only count
    exactly, ImageNette/CIFAR-10's existing behavior)."""
    penalty_genes = num_penalty_genes(config)
    if config.N_GENES == penalty_genes:
        return False
    if config.N_GENES == penalty_genes + 1:
        return True
    raise ValueError(
        f"config.N_GENES={config.N_GENES} doesn't match either the penalty-only "
        f"chromosome length ({penalty_genes}) or the penalty-plus-scheme length "
        f"({penalty_genes + 1}) for NUM_CLASSES={config.NUM_CLASSES}"
    )


def decode_scheme(chromosome, config):
    """Returns the weighting-scheme name ("INS"/"ISNS"/"ENS") a
    chromosome selects. Always "INS" for tracks that don't search the
    scheme (searches_weighting_scheme(config) is False) — the gene simply
    doesn't exist on those chromosomes, matching their historical,
    settled behavior exactly.

    The scheme gene is a continuous value in
    [config.SCHEME_GENE_LOWER_BOUND, config.SCHEME_GENE_UPPER_BOUND);
    floored and clamped into one of 3 equal-width bins, one per scheme,
    the simplest continuous-to-categorical mapping that gives pymoo's
    SBX/polynomial-mutation operators (which only ever produce continuous
    reals) a discrete 3-way choice to search over."""
    if not searches_weighting_scheme(config):
        return "INS"

    gene_value = chromosome[num_penalty_genes(config)]
    span = config.SCHEME_GENE_UPPER_BOUND - config.SCHEME_GENE_LOWER_BOUND
    fraction = (gene_value - config.SCHEME_GENE_LOWER_BOUND) / span
    index = int(np.clip(np.floor(fraction * len(WEIGHTING_SCHEMES)), 0, len(WEIGHTING_SCHEMES) - 1))
    return WEIGHTING_SCHEMES[index]

# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The ref-block banding spec of `mosaic ingest --ref-gq-bands`:
``RefBandSpec``, the parsed and normalized GQ band bounds.

``mosaic ingest`` parses the option with it (an argparse error for a
malformed spec), records the normalized text in
``ingestion_progress.ref_gq_bands`` (a resume must give the same value) and
passes it to the shard worker in the manifest. The banding itself (a
sample's adjacent ref blocks of one class and GQ band within one
``--pos-batch-size`` window merged into one ``sample_ref_blocks`` row: dp
the length-weighted mean rounded half up, AD / PL of the lowest-GQ block) is
the Rust worker's ``rust/mosaic-ingest-worker/src/banding.rs``, which parses
the same spec; the Python bander that lived here went with the earlier
Python ingest worker.
"""

import re
from dataclasses import dataclass
from typing import Tuple

from .genotypes import GQ_CAP

# The one-band spec: every adjacent block of the same class merges,
# whatever its (known) GQ.
REF_GQ_BANDS_ALL = "all"

_BOUND = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class RefBandSpec:
    """A GQ band spec: the bands' exclusive upper bounds, ascending, each in
    [1, GQ_CAP] (GATK ``--gvcf-gq-bands`` semantics). ``(20, 60)`` is the
    bands [0, 20), [20, 60), [60, 99]; ``()`` is one band (``all``)."""

    bounds: Tuple[int, ...]

    def __post_init__(self) -> None:
        for b in self.bounds:
            if not 1 <= b <= GQ_CAP:
                raise ValueError(f"GQ band bound {b} is outside [1, {GQ_CAP}]")
        if any(b <= a for a, b in zip(self.bounds, self.bounds[1:])):
            raise ValueError(
                "GQ band bounds must be strictly ascending (no repeat): "
                + ",".join(str(b) for b in self.bounds)
            )

    @classmethod
    def parse(cls, text: str) -> "RefBandSpec":
        """``20,60`` / ``all`` (spaces around the items ignored). Raises
        ValueError for an empty spec, a non-integer item, a bound outside
        [1, 99], unsorted or repeated bounds."""
        s = text.strip()
        if s == REF_GQ_BANDS_ALL:
            return cls(())
        if not s:
            raise ValueError(
                "an empty GQ band spec: give ascending bounds in "
                f"[1, {GQ_CAP}] (e.g. 20,60) or '{REF_GQ_BANDS_ALL}'"
            )
        bounds = []
        for item in s.split(","):
            item = item.strip()
            if not _BOUND.fullmatch(item):
                raise ValueError(
                    f"GQ band bound {item!r} is not an integer (spec "
                    f"{text!r}; give ascending bounds in [1, {GQ_CAP}] or "
                    f"'{REF_GQ_BANDS_ALL}')"
                )
            bounds.append(int(item))
        return cls(tuple(bounds))

    @property
    def text(self) -> str:
        """The normalized spec: ``20,60`` or ``all`` (what the banner prints
        and ``ingestion_progress.ref_gq_bands`` records)."""
        if not self.bounds:
            return REF_GQ_BANDS_ALL
        return ",".join(str(b) for b in self.bounds)

    def __str__(self) -> str:
        return self.text

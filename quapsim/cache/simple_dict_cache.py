#!/usr/bin/env python3

import numpy as np
from typing import List, Union

from .interface import ICache
from quapsim.gates import IGate
from .utils import TrieNode, create_key, add_to_trie


class SimpleDictCache(ICache):
    """Base class for all gate sequence caches."""

    def __init__(self):
        self._dict = {}
        self._trie_root = TrieNode(gate=None)

    def add(self, gate_sequence: List[IGate], unitary: np.ndarray) -> None:
        """Add the unitary of a sequence of gates to the cache."""
        self._add_to_trie(gate_sequence)
        self._add_to_cache(gate_sequence, unitary)

    def _add_to_trie(self, gate_sequence: List[IGate]) -> None:
        add_to_trie(gate_sequence, self._trie_root)

    def _add_to_cache(self, gate_sequence: List[IGate], unitary: np.ndarray) -> None:
        key = create_key(gate_sequence)
        self._dict[key] = unitary

    def get_prefix_in_cache_length(self, gate_sequence: List[IGate]) -> int:
        """Return the maximum sequence length that is in the cache,
        measured from the beginning of the sequence. This method is
        use full for performing cache lookup."""
        prefix_length = 0

        current_node = self._trie_root
        for i, gate in enumerate(gate_sequence):

            child = current_node.get_child(gate)

            # Case: current node has no children corresponding to
            # the gate currently under investigation.
            if child is None:
                break

            if child.is_sequence_end:
                prefix_length = i + 1

            current_node = child

        return prefix_length

    def get(self, gate_sequence: List[IGate]) -> Union[None, np.ndarray]:
        """Retrieve the unitary of a sequence of gates from the cache
        if it exists. Else return None."""
        key = create_key(gate_sequence)

        if key not in self._dict:
            return None
        else:
            return self._dict[key]

    def reset(self) -> None:
        """Clear the cache."""
        self._dict = {}

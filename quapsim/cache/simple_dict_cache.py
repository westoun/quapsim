#!/usr/bin/env python3

import numpy as np
from typing import List, Union, Set

from .interface import ICache
from quapsim.gates import IGate


def create_key(gates: List[IGate]) -> str:
    key = "_".join([gate.__repr__() for gate in gates])
    return key


class TrieNode:
    gate: IGate
    is_sequence_end: bool
    children: List["TrieNode"]

    def __init__(self, gate: IGate):
        self.gate = gate
        self.children = []
        self.is_sequence_end = False

    def get_child(self, gate: IGate) -> Union["TrieNode", None]:
        for child in self.children:
            if child.gate == gate:
                return child

        return None

    def add_child(self, node: "TrieNode") -> None:
        self.children.append(node)

    def __repr__(self, tabs: int = 0) -> str:
        representation: str = ""

        if self.gate is not None:
            for _ in range(tabs):
                representation += "\t"

            representation += self.gate.__repr__()

        for child in self.children:
            representation += "\n" + child.__repr__(tabs=tabs + 1)

        return representation


def add_to_trie(gate_sequence: List[IGate], node: TrieNode) -> None:
    if len(gate_sequence) == 0:
        node.is_sequence_end = True
        return

    child_node = node.get_child(gate_sequence[0])

    if child_node is None:
        child_node = TrieNode(gate_sequence[0])
        node.add_child(child_node)

    add_to_trie(gate_sequence[1:], child_node)


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

            for child in current_node.children:
                if child.gate == gate:
                    if child.is_sequence_end:
                        prefix_length = i + 1

                    current_node = child
                    break
            
            # Case: current node has no children corresponding to 
            # the gate currently under investigation.
            else: 
                break

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

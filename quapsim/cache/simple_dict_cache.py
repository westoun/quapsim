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
    children: List["TrieNode"]

    def __init__(self, gate: IGate):
        self.gate = gate
        self.children = []

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


def is_in_trie(gate_sequence: List[IGate], node: TrieNode) -> bool:
    if len(gate_sequence) == 0:
        return True

    child_node = node.get_child(gate_sequence[0])
    if child_node is None:
        return False

    return is_in_trie(gate_sequence[1:], child_node)


def add_to_trie(gate_sequence: List[IGate], node: TrieNode) -> None:
    if len(gate_sequence) == 0:
        return

    child_node = node.get_child(gate_sequence[0])

    if child_node is None:
        child_node = TrieNode(gate_sequence[0])
        node.add_child(child_node)

    add_to_trie(gate_sequence[1:], child_node)


class Trie:

    def __init__(self):
        self._trie_root = TrieNode(gate=None)

    def contains(self, gate_sequence: List[IGate]) -> bool:
        return is_in_trie(gate_sequence, self._trie_root)

    def add(self, gate_sequence: List[IGate]) -> bool:
        add_to_trie(gate_sequence, self._trie_root)

    def __repr__(self) -> str:
        return self._trie_root.__repr__()


class SimpleDictCache(ICache):
    """Base class for all gate sequence caches."""

    def __init__(self):
        self._dict = {}
        self._trie = Trie()

    def could_be_in_cache(self, gate_sequence: List[IGate]) -> bool:
        """Return if a gate sequence or its children could be in
        the cache (based on internal trie structure).
        """
        return self._trie.contains(gate_sequence)

    def add(self, gate_sequence: List[IGate], unitary: np.ndarray) -> None:
        """Add the unitary of a sequence of gates to the cache."""
        self._add_to_trie(gate_sequence)
        self._add_to_cache(gate_sequence, unitary)

    def _add_to_trie(self, gate_sequence: List[IGate]) -> None:
        self._trie.add(gate_sequence)

    def _add_to_cache(self, gate_sequence: List[IGate], unitary: np.ndarray) -> None:
        key = create_key(gate_sequence)
        self._dict[key] = unitary

    def get(self, gate_sequence: List[IGate]) -> Union[None, np.ndarray]:
        """Retrieve the unitary of a sequence of gates from the cache
        if it exists. Else return None."""
        key = create_key(gate_sequence)

        if key not in self._dict:
            return None
        else:
            return self._dict[key]

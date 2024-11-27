#!/usr/bin/env python3

import numpy as np
from typing import List, Union, Set, Tuple

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


def is_in_trie(gate_sequence: List[IGate], node: TrieNode) -> bool:
    """The first boolean of the returned tuple answers whether the
    sequence is contained in the trie. The second one answers if
    the final node is a sequence end."""

    if len(gate_sequence) == 0:
        return True, node.is_sequence_end

    child_node = node.get_child(gate_sequence[0])
    if child_node is None:
        return False, False

    return is_in_trie(gate_sequence[1:], child_node)


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

    def get_cache_potential(self, gate_sequence: List[IGate]) -> Tuple[bool, bool]:
        """The first boolean of the returned tuple answers whether the
        any child sequences could be contained in the cache.
        The second one answers if it actually is."""
        return is_in_trie(gate_sequence, self._trie_root)

    def add(self, gate_sequence: List[IGate], unitary: np.ndarray) -> None:
        """Add the unitary of a sequence of gates to the cache."""
        self._add_to_trie(gate_sequence)
        self._add_to_cache(gate_sequence, unitary)

    def _add_to_trie(self, gate_sequence: List[IGate]) -> None:
        add_to_trie(gate_sequence, self._trie_root)

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

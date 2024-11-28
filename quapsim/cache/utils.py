#!/usr/bin/env python3

from typing import List, Union, Dict

from quapsim.gates import IGate


def create_key(gates: List[IGate]) -> str:
    key = "_".join([gate.__repr__() for gate in gates])
    return key


class TrieNode:
    gate: IGate
    is_sequence_end: bool
    children: Dict

    def __init__(self, gate: IGate):
        self.gate = gate
        self.children = {}
        self.is_sequence_end = False

    def get_child(self, gate: IGate) -> Union["TrieNode", None]:
        if gate in self.children:
            return self.children[gate]

        return None

    def add_child(self, node: "TrieNode") -> None:
        self.children[node.gate] = node

    def __repr__(self, tabs: int = 0) -> str:
        representation: str = ""

        if self.gate is not None:
            for _ in range(tabs):
                representation += "\t"

            representation += self.gate.__repr__()
        else:
            representation += "ROOT"

        for child in self.children.values():
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

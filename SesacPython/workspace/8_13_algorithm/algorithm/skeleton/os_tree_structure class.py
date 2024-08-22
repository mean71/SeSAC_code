import os 

# from ADT.tree import Tree 
from data_structure.tree import Tree

# def get_directory_tree(directory, ignore_directories = [], ignore_extensions = []):
#     pass 
class TreeNode:
    def __init__(self, id, data):
        self.id = id
        self.data = data

class Tree:
    def __init__(self, root, children = []):
        if not isinstance(root, TreeNode):
            root  = TreeNode([], root)
        self.root = root
        self.children = [Tree( TreeNode(str(idx),child) ) for idx,child in enumerate(children)]
        

        
if __name__ == '__main__':
    # print(get_directory_tree('.'))
    folder = Tree('./', [
        'bank_simulation.py',
        'data_structure/', [
            'graph.py',
            'graph_datastructure.py',
            'linked_list.py',
            'node.py',
            'structure_tree.py',
            '__pycache__/', [
                'linked_list.cpython-312.pyc',
                'node.cpython-312.pyc'
            ])
        ],
        'formula.py',
        'global_variables.py',
        'os_tree_structure.py',
        'os_tree_structure class',
        'sorting_8_12/', [
            'measure_performance.py',
            'sorting.py'
        ]),
        Tree('subway_map.py')
    ])



"""
./
├── bank_simulation.py
├── data_structure/
│   ├── graph.py
│   ├── graph_datastructure.py
│   ├── linked_list.py
│   ├── node.py
│   ├── structure_tree.py
│   └── __pycache__/
│       ├── linked_list.cpython-312.pyc
│       └── node.cpython-312.pyc
├── formula.py
├── global_variables.py
├── os_tree_structure.py
├── os_tree_structure class
├── sorting_8_12/
│   ├── measure_performance.py
│   └── sorting.py
└── subway_map.py
"""
"""
./
├── ADT/
│   ├── ADT_graph.py
│   ├── ADT_tree.py
│   ├── queue.py
│   ├── stack.py
│   └── __pycache__/
│       └── linked_list.cpython-312.pyc
├── bank_simulation.py
├── data_structure/
│   ├── graph.py
│   ├── graph_datastructure.py
│   ├── linked_list.py
│   ├── node.py
│   ├── structure_tree.py
│   └── __pycache__/
│       ├── linked_list.cpython-312.pyc
│       └── node.cpython-312.pyc
├── formula.py
├── global_variables.py
├── os_tree_structure.py
├── os_tree_structure class
├── sorting_8_12/
│   ├── measure_performance.py
│   └── sorting.py
└── subway_map.py
"""
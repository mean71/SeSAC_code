import os 

from ADT.tree import Tree 
from data_structure.tree import Tree

def get_directory_tree(directory, ignore_directories = [], ignore_extensions = []):
    pass 


if __name__ == '__main__':
    print(get_directory_tree('.'))

"""
./
├── ADT/
│   ├── queue.py
│   ├── stack.py
│   ├── tree.py
│   └── __pycache__/
│       ├── queue.cpython-39.pyc
│       └── stack.cpython-39.pyc
├── bank_simulation.py
├── data_structure/
│   ├── linked_list.py
│   ├── node.py
│   ├── tree.py
│   └── __pycache__/
│       ├── linked_list.cpython-39.pyc
│       ├── node.cpython-39.pyc
│       └── tree.cpython-39.pyc
├── formula.py
├── global_variables.py
├── os_tree_structure.py
├── sorting/
│   ├── experiment result/
│   │   ├── merge_sort.png
│   │   ├── quick_sort.png
│   │   ├── sort3_insert.png
│   │   ├── sort3_insert_1000.png
│   │   └── sorted.png
│   ├── measure_performance.py
│   ├── sorting.py
│   └── __pycache__/
│       └── sorting.cpython-39.pyc
├── util.py
└── __pycache__/
    ├── global_variables.cpython-39.pyc
    └── util.cpython-39.pyc
"""
import sys 
sys.path.append('../data_structure')

try:
    from linked_list import LinkedList, LinkedNode, DoublyLinkedNode, DoublyLinkedList
    from tree import Tree as TreeDataStructure
except ModuleNotFoundError:
    from data_structure.linked_list import LinkedList, LinkedNode, DoublyLinkedNode, DoublyLinkedList
    from data_structure.tree import Tree as TreeDataStructure


class Tree:
    def __init__(self, tree_elements, backend):
        """Intialize Tree from tree_elements dict. 

        Examples
            t1 = Tree({1: [
                        {2: []}, 
                        {3: []}
                        ]})
            print(t1)
            >> 1
            ├── 2
            └── 3
            t2 = Tree({1: [{11, [{111:[]}, {112:[]},], 
                           {12, [{121:[]}]]})

            print(t2)
            >> 1
            ├── 11
            │   ├── 111
            │   └── 112
            └── 12
                └── 121

        """
        self.backend = backend
        
        if backend == TreeDataStructure:
            pass 

    def insert(self, address, elem):
        pass 

    def delete(self, address):
        pass 

    def search(self, elem):
        pass 

    def root(self):
        pass 

    def height(self):
        pass 

    def __str__(self):
        pass 


if __name__ == '__main__':
    t1 = Tree({1: [
                    {
                        11: [
                            {111:[]}, 
                            {112:[]},
                        ]
                    }, 
                    {
                        12: [
                            {121:[]},
                            {122:[]},
                            {123:[]},
                        ]
                    },
                ]})


    print(t1)

    assert t1.root() == 1 
    assert t1.height() == 3

    t1.insert([2], 13)
    t1.insert([2, 0], 131)
    t1.insert([2, 1], 132)
    t1.insert([2, 2], 133)
    t1.insert([1, 1], 122)
    t1.insert([1, 1, 0], 1221)
    t1.insert([1, 1, 1], 1222)

    print(t1)

    assert 122 == t1.search([1,2])
    assert 122 == t1.delete([1,2])
    assert 123 == t1.search([1,2])
    assert 123 == t1.delete([1,2])

    print(t1)

    


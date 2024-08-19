class TreeNode:
    def __init__(self, node_id, datum):
        self.node_id = node_id
        self.datum = datum 

class Tree:
    def __init__(self, root, children = []):
        if not isinstance(root, TreeNode):
            root = TreeNode('0', root)
        self.root = root 
        
        children = list(children)
        for idx, child in enumerate(children):
            if not isinstance(child, Tree):
                children[idx] = Tree(root = TreeNode(str(idx), child))
            
        self.children = children 

    def iter_nodes(self):
        yield self.root 

        for child in self.children:
            for n in child.iter_nodes():
                yield n 

    def iter_nodes_with_address(self):
        yield [], self.root 

        for idx, child in enumerate(self.children):
            for addr, n in child.iter_nodes_with_address():
                yield [idx] + addr, n 

    def __iter__(self):
        yield self.root.datum

        for child in self.children:
            for n in child.iter_nodes():
                yield n 

    def insert(self, address, elem):
        if not isinstance(elem, Tree):
            elem = Tree(elem) 

        cur = self 
        for addr in address[:-1]:
            cur = cur.children[addr]
        cur.children.insert(address[-1], elem)

    def delete(self, address):
        cur = self 
        
        for addr in address[:-1]:
            cur = cur.children[addr] 

        res = cur.children[address[-1]].root.datum 
        del cur.children[address[-1]]

        return res 
        
    def search(self, elem):
        for addr, node in self.iter_nodes_with_address():
            if node.datum == elem:
                return addr 

    def root_datum(self):
        return self.root.datum 

    def height(self):
        h = 0 

        for addr, node in self.iter_nodes_with_address():
            if len(addr) + 1 > h:
                h = len(addr) + 1

        return h 

    def __str__(self):
        res = str(self.root.datum) 

        for idx, child in enumerate(self.children):
            res += '\n'
            if idx < len(self.children) - 1:
                res += '├── '
                res += str(child).replace('\n', '\n│   ')
            else:
                res += '└── '
                res += str(child).replace('\n', '\n    ')
        
        return res 


if __name__ == '__main__':
    t1 = Tree(1, [
                Tree(11, [Tree(111), Tree(112)],), 
                Tree(12, [Tree(121), Tree(122), Tree(123),])
             ]
         )
    print(t1)
    
    assert t1.root_datum() == 1 
    assert t1.height() == 3

    for addr, n in t1.iter_nodes_with_address():
        assert [int(e)-1 for e in list(str(n.datum))[1:]] == addr 
        assert t1.search(n.datum) == addr 

    t1.insert([2], Tree(13, [Tree(131), Tree(132), Tree(133)]))
    t1.insert([1, 1], Tree(122, [Tree(1221), Tree(1222)]))

    print(t1)
    
    assert 122 == t1.delete([1,2])
    assert 123 == t1.delete([1,2])

    for addr, n in t1.iter_nodes_with_address():
        assert [int(e)-1 for e in list(str(n.datum))[1:]] == addr 
        assert t1.search(n.datum) == addr 

    print(t1)
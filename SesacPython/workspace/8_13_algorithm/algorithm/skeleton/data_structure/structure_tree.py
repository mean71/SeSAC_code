try:
    from node import Node 
except ModuleNotFoundError:
    from data_structure.node import Node

class TreeNode:
    def __init__(self, node_id, datum):
        self.node_id = node_id
        self.datum = datum

# class tree:
#     def __init__(self, data):
#         self.data = data
#         self.child = []
#         self.parents = None
#     def new_node(self, new_node):
#         new_node.parents = self
#         self.child.append(new_node)
# root = tree('프로젝트폴더')

# python = tree('python')
# python.new_node(tree('python1'))
# python.new_node(tree('python2'))
# root.new_node(python)

class Tree:
    def __init__(self, root, children = []):# root로 받은 인자가 노드 인스턴스가 아니라면 주소'0'인 노드인스턴스로 만들어주고 root로 지정
        self.root = TreeNode('0',root)
        children = list(children)
        for idx,child in enumerate(children):
            children[idx] = Tree(TreeNode(str(idx), child))
        # 자식트리로 받은 요소를 리스트로 변환 -> 반복문으로 주소 인덱스와 datum 으로 분리해서 그대로 노드인스턴스로 변환하여 리스트에 다시 대입. # 트리노드인스턴스 리스트를 자식트리로 저장

    def iter_nodes(self):

        pass

    def iter_nodes_with_address(self):
        pass

    def __iter__(self):
        pass 

    def insert(self, address, elem):
        pass 

    def delete(self, address):
        pass
        
    def search(self, elem):
        pass 

    def root_datum(self):
        pass 

    def height(self):
        pass 

    def __str__(self):
        return '미완성' 


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
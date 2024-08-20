class TreeNode:
    def __init__(self, node_id, datum):
        self.node_id = node_id
        self.datum = datum 

class Tree: # 트리는 이중연결하면 삑난다.
    def __init__(self, root, children = []):
        if not isinstance(root, TreeNode): # root로 받은 인자가 노드 인스턴스가 아니라면 주소'0'인 노드인스턴스로 만들어주고 root로 지정
            root = TreeNode('0', root)
        self.root = root
        
        children = list(children) # 자식트리로 받은 요소를 리스트로 변환 -> 반복문으로 주소 인덱스와 datum 으로 분리해서 그대로 노드인스턴스로 변환하여 리스트에 다시 대입.
        for idx, child in enumerate(children):
            if not isinstance(child, Tree):
                children[idx] = Tree(root = TreeNode(str(idx), child)) # Tree(root = TreeNode(str(idx), child)) 인 이유를 정확히 이해
            
        self.children = children # 트리노드인스턴스 리스트를 자식트리로 저장

    def iter_nodes(self):   # 노드 순회 : 명칭은? # root 노드반환 # 자식트리 모든노드를 순회
        yield self.root # 루트노드 반환

        for child in self.children: # 자식트리노드 리스트를 순회
            for n in child.iter_nodes(): # 이게 어찌 이렇게도...
                yield n 

    def iter_nodes_with_address(self):  
        yield [], self.root # 루트 노드주소[], 노드 반환, 이어서 실행

        for idx, child in enumerate(self.children): # 노드배열을 enumerate로 묶어 반복 다시 자기자신과 반복
            for addr, n in child.iter_nodes_with_address():
                yield [idx] + addr, n

    def __iter__(self):
        # print(f'__iter__ called with {self}')
        yield self.root.datum # 노드 .data를 반환 이어서 노드 순회하며 자식노드.data 반환

        for child in self.children:
            for n in child: #child.iter_nodes():
                yield n

    def insert(self, address, elem): # idx와 요소를 받아서 
        if not isinstance(elem, Tree):
            elem = Tree(elem) 

        cur = self # 자신을 참조하고 자신의 자식노드[idx] address에 도달할때까지 self.children 반복순회 idx[:-1] 클래스자신의 자식노드리스트의 끝에 새로이 TreeNode(elem)추가
        for addr in address[:-1]:
            cur = cur.children[addr]
        cur.children.insert(address[-1], elem)

    def delete(self, address):
        cur = self # 자기 자신을 참조하고 자신의 자식노드 address에 도달할때까지 self.children 반복순회 반대로 cur.children[address[-1]]노드를 del함수로 제거
        
        for addr in address[:-1]:
            cur = cur.children[addr] 

        res = cur.children[address[-1]].root.datum # 제거한 노드의.root.datum을 반환
        del cur.children[address[-1]]

        return res
        
    def search(self, elem):
        for addr, node in self.iter_nodes_with_address(): # iter_nodes_with_address 에서 주소와 노드를 받아 반복 node.datum 을 비교해서 받은 인자와 일치하면 주소를 찾아 반환
            if node.datum == elem:
                return addr 

    def root_datum(self):
        return self.root.datum

    def height(self):
        h = 0
        for addr, _ in self.iter_nodes_with_address(): # 트리의 size?높이 leaf노드와 root 노드간의 연결거리 # 반환값h로 값을 초기화하고 iter_nodes_with_addressd를 순환
            if len(addr) + 1 > h: # 현재순환중인 노드의len(idx)+가 h 보다 크다면 h에 대입 # 트리의 높이를 반환
                h = len(addr) + 1   
        return h

    def __str__(self):
        res = str(self.root.datum) # root.datum을 반환값으로!

        for idx, child in enumerate(self.children): # 자식노드를  idx와 같이 순환 한번 순환될때마다 줄이바뀌며 반환될 문자열이 추가된다. # WOW.....................
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

    for e in t1:
        print(e)
    
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
    print('===============')
    for addr, n in t1.iter_nodes_with_address():
        print(addr, n.datum)
    print('===============')
    for addr, n in t1.children[1].iter_nodes_with_address():
        print(addr, n.datum)
    # for i in t1.iter_nodes():
    #     print(i)
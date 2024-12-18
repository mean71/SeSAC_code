try:
    from node import Node
except ModuleNotFoundError:
    from data_structure.node import Node


class TreeNode:
    def __init__(self, node_id, datum):
        self.node_id = node_id  # idx 주소로 활용
        self.datum = datum  # 자식트리의 손자트리 노드?로 활용 # 조금 뇌정지


class Tree:
    def __init__(self, root, children=[]):  # 자식트리가없으면 기본값[]
        if not isinstance(root, TreeNode):
            root = TreeNode([], root)
        self.root = root
        children = list(children)

        self.children = children = [
            TreeNode(str(idx), child) for idx, child in enumerate(children)
        ]  # iter_nodes(): child.root가 왜 새 방식에서만 활성화되는가는 졸리고 귀찮으니 나중에
        # [ Tree(TreeNode(str(idx), child)), ... ,  ] head.next 와 같다.  모든 자식트리가 다시 부모가 되며 반복된다.
        # ***자식트리를 리스트로 넣고 for문으로 순환시키면서 linkedlist 비스무리하게 재귀적?으로 우려먹는 과정을 숙지***

    def iter_nodes(
        self,
    ):  # 노드만 iterator로 반환 # 클래스인스턴스의 재귀에 대한 반쪽짜리 이해
        yield self.root  # Tree의 self.root 를 반환하고 자식트리들을 순회하면서 재귀로 다시 yield self.root 반환, 또 그 자식트리를 순환코드가 실행되고 자식트리가 없어서 멈추고 돌아갈때까지 노드 반환을 반복
        for child in self.children:
            for i in child.iter_nodes():
                yield i
            # yield child.root # 이러면... 자식노드의 하위자식이.. # 트리구현 예제실습

    def iter_nodes_with_address(self):  # idx, node
        yield [], self.root  # 첫 idx는 []로 root만 뱉고
        for idx, child in enumerate(
            self.children
        ):  # 마찬가지로 인덱스[0,1,...]와 같이 동일하게 순회
            for (
                i,
                n,
            ) in (
                child.iter_nodes_with_address()
            ):  # child는 Tree(TreeNode(str(idx), child))
                yield [idx] + i, n  # [idx,i,i,i...], root값

    def __iter__(self):
        yield self.root.datum
        for child in self.children:
            for i in child:
                yield i  # i.datum 문제발생 졸려서 보류하고 패스

    # def insert(self, address, elem):
    #     cur =self
    #     elem = TreeNode(elem)
    #     cur = cur.children # child 순환돌려서 idx 일치하면 children.insert([idx], elem)박아넣기?
    #     address

    def delete(
        self, address
    ):  # child 순환돌려서 idx 일치하면 del tree_lst[idx] 이후  .datum 반환 # t1.delete([1,2])
        res = []
        for (
            idx,
            j,
        ) in self.iter_nodes_with_address():  # 이러면 자식트리의 자식트리수정이 [1,2]
            if address == idx:
                del self.children[idx]
                return f"{j} delete"

    def search(self, elem):  # 순환돌려서 .datum 이 동일하면 idx 반환 방법은 ...
        for idx, j in self.iter_nodes_with_address():
            if elem == j.datum:  # .datum
                return idx

    def root_datum(self):
        return self.root.datum

    def height(self):

        h = 0
        for i, n in self.iter_nodes_with_address():
            if h < len(i) + 1:
                h = len(i) + 1  # 높이층위
            print(i, n)
        return h  # root 에서는 [] ,h=0, iter_nodes_with_address에서 idx리스트가 하나씩 추가, for child 돌려서 max(len([idx]))를 반환?

    def __str__(self):
        res = str(self.root.datum)

        for (
            child
        ) in (
            self.children
        ):  # child 또한 Tree(TreeNode(str(idx), child)) # 자식 트리의 __str__은 알아서 다시 반복 # for child in self.children: 로하면
            res += "\n"
            if child != self.children[-1]:  # 일단 이대로 실행
                res += "├── "
                if child.root:
                    res += str(child).replace(
                        "\n", "\n│   "
                    )  # 본인을 출력하고 자식트리가 있다면 또다시 줄바꿈하고 같은것을 반복
            else:
                res += "└── "
                if child.root:
                    res += str(child).replace("\n", "\n    ")
        return res


if __name__ == "__main__":
    t1 = Tree(
        1,
        [
            Tree(
                11,
                [Tree(111, [Tree(1111, [Tree(11111)])]), Tree(112)],
            ),
            Tree(
                12,
                [
                    Tree(121),
                    Tree(122),
                    Tree(123),
                ],
            ),
            # Tree(13, [Tree(131, [Tree(1311), Tree(1312)],)]) # 중복된 노드 삽입을 insert에서 방지하지 않으면 에러 발생# insert, delete에서 잘못된 주소를 추가할때의 예외사항도 필요
        ],
    )
    print(t1)
    for i, t in t1.iter_nodes_with_address():
        print(i)
        print(t.datum)

    for e in t1:
        print(e)

    assert t1.root_datum() == 1
    assert t1.height() == 5

    for addr, n in t1.iter_nodes_with_address():  # [], node 순회
        print(list(str(n.datum))[1:])
        assert [
            int(e) - 1 for e in list(str(n.datum))[1:]
        ] == addr  # node.datum과 node.id = [idx]가 양식에 맞는지 체크 # 용도와 의도가 해석이 되지만 피곤해서 논리와 글로 정리하고 사고하는게 불가능
        assert t1.search(n.datum) == addr

    # t1.insert([2], Tree(13, [Tree(131), Tree(132), Tree(133)]))
    # t1.insert([1, 1], Tree(122, [Tree(1221), Tree(1222)]))
    # print(t1)

    # assert 122 == t1.delete([1,2])
    # assert 123 == t1.delete([1,2])
    # assert 1111 == t1.delete([0,0,0])
    # assert t1.height == 4

    # for addr, n in t1.iter_nodes_with_address():
    #     assert [int(e)-1 for e in list(str(n.datum))[1:]] == addr
    #     assert t1.search(n.datum) == addr
    # print(t1)

    print("===============")
    for addr, n in t1.iter_nodes_with_address():
        print(addr, n.datum)
    print("===============")
    for addr, n in t1.children[1].iter_nodes_with_address():
        print(addr, n.datum)
    for i in t1.iter_nodes():
        print(i)


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

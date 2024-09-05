import os

# from ADT.tree import Tree
# from data_structure.tree import Tree


# def get_directory_tree(directory, ignore_directories = [], ignore_extensions = []):
#     pass
class TreeNode:
    def __init__(self, id, data):
        self.id = id
        self.data = data


class Tree:
    def __init__(self, root, children=[]):
        if not isinstance(root, TreeNode):
            root = TreeNode([], root)
        self.root = root  # root는 단하나뿐인 부모TreeNode인스턴스  

        self.children = []
        children = list(children)
        for idx, child in enumerate(children):
            if len(child) == 2:
                self.children.append(Tree(root=TreeNode([str(idx)], child[0]), children=child[1]))
            else:
                self.children.append(Tree(root=TreeNode([str(idx)], child)))  # iter_nodes_with_address 반환주소.id를 [0,1] .data="파일명"
        # 자식들은 트리의 인스턴스 노드로 변환, (idx) TreeNode .id .data로 # 이또한 children = tree(root,children[])들의 리스트 라는 구조에 대한 명확한 인지가 필요
        # self.children = children 으로 할시 문자열이 해체되어 출력 되는 과정도 해석해보기

    def iter_nodes(self):  # TreeNode순회
        yield self.root
        for child in self.children:  # children의 요소들은 하나하나가 자식트리
            for n in child:
                yield n
        # for n in self.iter_nodes: print(n.data)

    def iter_nodes_with_address(self):  # idx로 엮어서 노드위치 값 생성 TreeNode.id , root = TreeNode(id,data) 순회
        yield self.root.id, self.root
        for idx, child in enumerate(self.children):
            for i, n in child.iter_nodes_with_address():
                yield [str(idx)] + i, n  #  # yield [idx] + child.root.id, n # 이따구로 하면 안될듯하다. # n.root.data 될리가.

    def __iter__(self):  # TreeNode.data순회
        yield self.root.data  #'\\' 예시 : skeleton\os_tree_structure class.py
        for child in self.children:
            for n in child:
                yield n  # yield n.root.data # 이따구로 해도 안된다.

    def insert(self, address, elem):

        pass

    def delete(self, address):

        pass

    def search(self, elem):

        pass

    def root_datum(self):
        return self.root.data

    def height(self):
        h = 0
        for idx, _ in self.iter_nodes_with_address():
            if h <= len(idx):
                h = len(idx) + 1
            return h

    def __str__(self):
        res = str(self.root.data)
        for child in self.children:  # child = Tree(TreeNode,[...]) 즉 반복문 안에서 __str__의 출력으로 들어가면? 자식트리가 []가 될때까지 재귀가 시작된다.
            res += "\n"  # __str__이라 "문자열로 호출"될때 한정. # 문자열로 만들어줘야 재귀가 된다. str(child)
            if child != self.children[-1]:
                res += "├── "
                res += str(child).replace("\n", "\n│   ")
            else:
                res += "└── "
                res += str(child).replace("\n", "\n     ")
        return res

    def s(t):  # 서비스 문제
        """
        ./
            bank_simulation.py
            data_structure/
                graph.py
                graph_datastructure.py
                linked_list.py
                node.py
                structure_tree.py
                __pycache__/
                    linked_list.cpython-312.pyc
                    node.cpython-312.pyc
            formula.py
            global_variables.py
            os_tree_structure.py
            os_tree_structure class
            sorting_8_12/
                measure_performance.py
                sorting.py
            subway_map.py
        """
        pass


if __name__ == "__main__":
    # print(get_directory_tree('.'))
    dr = Tree(
        "skeleton/",
        [
            "bank_simulation.py",
            (
                "data_structure/",
                [
                    "graph.py",
                    "graph_datastructure.py",
                    "linked_list.py",
                    "node.py",
                    "structure_tree.py",
                    (
                        "__pycache__/",
                        [
                            "linked_list.cpython-312.pyc",
                            "node.cpython-312.pyc"
                        ],
                    ),
                ],
            ),
            "formula.py",
            "global_variables.py",
            "os_tree_structure.py",
            "os_tree_structure class",
            ("sorting_8_12/",
              ["measure_performance.py", "sorting.py"]
              ),
            "subway_map.py",
        ],
    )

    print("\n print(dr)")
    print(dr)

    print("\n for i in dr: print(i)")
    for i in dr:
        print(i)

    print("\n for i in dr.iter_nodes_with_address(): print(adr, data.data)")
    for adr, data in dr.iter_nodes_with_address():
        print(adr, data.data)

    print("\n print(dr.height()) : ", dr.height())

    # print()
    # dr.s(?)


"""
skeleton/
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

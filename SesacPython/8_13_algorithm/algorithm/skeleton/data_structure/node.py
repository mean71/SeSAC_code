class Node:
    def __init__(self, node_id, datum):
        self.node_id = node_id 
        self.datum = datum 

    def __str__(self): # 클래스 출력시 노드id를 출력
        return f'node {self.node_id}'
# 노드기능을 담당하는 클래스 Node()
if __name__ == '__main__':
    assert str(Node(1)) == 'node 1'
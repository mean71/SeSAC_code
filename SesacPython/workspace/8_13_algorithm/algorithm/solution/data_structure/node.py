class Node:
    def __init__(self, node_id, datum):
        self.node_id = node_id 
        self.datum = datum 

    def __str__(self):
        return f'node {self.node_id}'

if __name__ == '__main__':
    assert str(Node(1)) == 'node 1'
try:
    from node import Node
except ModuleNotFoundError:
    from data_structure.node import Node

class LinkedNode:
    def __init__(self, node_id, datum, next = None): # 식별자, 데이터 다음노드
        self.node_id = node_id 
        self.datum = datum
        self.next = next # 기본값 None

# elements가 LinkedNode 클래스의 리스트
# 
class LinkedList:
    def __init__(self, elements):
        if elements == []: # 인자로 받은 리스트가 비었을때
            self.head = None
            self.tail = None
            self.end = None
            self.size = 0
        else:
            self.head = elements[0] # 리스트의 첫 노드
            self.tail = elements[-1] # 마지막 노드
            self.end = elements[-1] # 현재남은 리스트의 마지막? None이 나올때까지???
            self.size = len(elements) # 리스트사이즈
            for i in range(len(elements)-1):
                elements[i].next = elements[i+1] # 인자로 받은 리스트에서 요소를 호출해도 인스턴스 클래스의 인스턴스는 클래스의 속성을 공유가능하다.
                
                
                


                

    def __iter__(self): #
        
        yield None

    def __str__(self):

        return res

    def append(self, elem):
        if not isinstance(elem, LinkedNode):
            elem = LinkedNode(self.size, elem, next = None)

        self.end.next = elem
        self.end = elem
        self.size += 1



class DoublyLinkedNode(Node):
    def __init__(self, node_id, datum, prev = None, next = None):
        self.node_id = node_id
        self.datum = datum
        self.next = next
        self.prev = prev

class DoublyLinkedList:
    def __init__(self, elements):
        if elements == []:
            self.head = None
            self.tail = None
            self.end = None
            self.size = 0
        else:
            self.head = None
            self.tail = None
            self.end = None
            self.size = 0

    def __iter__(self):
        yield None

    def __str__(self):
        res = ''

        return res
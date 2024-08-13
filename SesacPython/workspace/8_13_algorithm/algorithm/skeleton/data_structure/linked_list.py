try:
    from node import Node 
except ModuleNotFoundError:
    from data_structure.node import Node

class LinkedNode:
    def __init__(self, node_id, datum, next = None):
        self.node_id = node_id
        self.datum = datum
        self.next = next

class LinkedList:
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


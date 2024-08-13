from node import Node 

class LinkedNode(Node):
    def __init__(self, node_id, datum, next = None):
        super().__init__(node_id, datum) 
        self.next = next 

class LinkedList:
    def __init__(self, elements):
        if elements == []:
            self.head = None 
            self.tail = None 
            self.end = None
            self.size = 0
        else:
            size = 0
            for idx, e in enumerate(elements):
                assert isinstance(e, LinkedNode)
                if idx < len(elements) - 1:
                    e.next = elements[idx+1]
                size += 1
            
            head = elements[0]
            tail = LinkedList(elements[1:])
            end = elements[-1]

            assert isinstance(tail, LinkedList) or tail is None 
            assert end.next is None
            self.head = head 
            self.tail = tail 
            self.end = end 
            self.size = size 

    def __iter__(self):
        cur = self.head

        while cur is not None:
            yield cur 
            cur = cur.next 

    def __str__(self):
        cur = self.head 
        res = ''

        while cur is not None:
            if cur == self.head:
                res += f'[head]->[{cur}]'
            else:
                res += f'->[{cur}]'
            cur = cur.next 
        
        res += f'->[None]'

        return res 

class DoublyLinkedNode(Node):
    def __init__(self, node_id, datum, prev = None, next = None):
        super().__init__(node_id, datum) 
        self.next = next 

class DoublyLinkedList:
    def __init__(self, elements):
        if elements == []:
            self.head = None 
            self.tail = None 
            self.end = None
            self.size = 0
        else:
            size = 0
            for idx, e in enumerate(elements):
                assert isinstance(e, DoublyLinkedNode)
                if idx < len(elements) - 1:
                    e.next = elements[idx+1]
                if 0 < idx:
                    e.prev = elements[idx-1]
                size += 1
            
            head = elements[0]
            tail = DoublyLinkedList(elements[1:])
            end = elements[-1]

            assert isinstance(tail, DoublyLinkedList) or tail is None 
            assert end.next is None

            self.head = head 
            self.tail = tail 
            self.end = end 
            self.size = size

    def __iter__(self):
        cur = self.head

        while cur is not None:
            yield cur 
            cur = cur.next 

    def __str__(self):
        cur = self.head 
        res = ''

        while cur is not None:
            if cur == self.head:
                res += f'[head]->[{cur}]'
            else:
                res += f'->[{cur}]'
            cur = cur.next 
        
        res += f'->[None]'

        return res 


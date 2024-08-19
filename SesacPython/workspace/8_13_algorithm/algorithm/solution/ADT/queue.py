import sys 
sys.path.append('../data_structure')

from linked_list import LinkedList, LinkedNode, DoublyLinkedNode, DoublyLinkedList

class Queue:
    def __init__(self, *elements, backend = list):
        self.backend = backend

        assert isinstance(elements, list) or isinstance(elements, tuple)
        
        if self.backend == LinkedList:
            res = []
            for idx, elem in enumerate(elements):
                res.append(LinkedNode(idx, elem))            
            self.linked_list = LinkedList(res)
        elif self.backend == DoublyLinkedList:
            res = []
            for idx, elem in enumerate(elements):
                res.append(DoublyLinkedNode(idx, elem))            
            self.doubly_linked_list = DoublyLinkedList(res)
        elif self.backend == list:
            self.list = list(elements)

    def elements(self):
        if self.backend == LinkedList:
            res = []
            cur = self.linked_list.head 

            while cur is not None:
                res.append(cur.datum)
                cur = cur.next 

            return res 
        elif self.backend == DoublyLinkedList:
            res = []
            cur = self.doubly_linked_list.head 

            while cur is not None:
                res.append(cur.datum)
                cur = cur.next 

            return res 
        elif self.backend == list:
            return self.list 

    def enqueue(self, elem):
        if self.backend == LinkedList:
            n = LinkedNode(self.linked_list.size, elem, None)
            if self.linked_list.size != 0:
                self.linked_list.end.next = n
                self.linked_list.end = n
                self.linked_list.size += 1 
            else:
                self.linked_list = LinkedList([n])
        elif self.backend == DoublyLinkedList:
            n = DoublyLinkedNode(self.doubly_linked_list.size, elem, None, None)
            if self.doubly_linked_list.size != 0:
                self.doubly_linked_list.end.next = n
                n.prev = self.doubly_linked_list.end 
                self.doubly_linked_list.end = n
                self.doubly_linked_list.size += 1 
            else:
                self.doubly_linked_list = DoublyLinkedList([n])
        elif self.backend == list:
            self.list.append(elem)

    def dequeue(self):
        if self.backend == LinkedList:
            for elem in self.linked_list:
                if elem.next == self.linked_list.end:
                    elem.next = None 
                    res = self.linked_list.end
                    self.linked_list.end = elem
                    self.linked_list.size -= 1 
            return res.datum 
        elif self.backend == DoublyLinkedList:
            res = self.doubly_linked_list.end
            assert self.doubly_linked_list.end.next is None
            self.doubly_linked_list.end = self.doubly_linked_list.end.prev
            self.doubly_linked_list.end.next = None 
            self.doubly_linked_list.size -= 1
            return res.datum
        elif self.backend == list:
            return self.list.pop()
                
    def front(self):
        if self.backend == LinkedList:
            return self.linked_list.end.datum
        elif self.backend == DoublyLinkedList:
            return self.doubly_linked_list.end.datum
        elif self.backend == list:
            return self.list[-1]

    def size(self):
        if self.backend == LinkedList:
            return self.linked_list.size
        elif self.backend == DoublyLinkedList:
            return self.doubly_linked_list.size 
        elif self.backend == list:
            return len(self.list)
    
    def is_empty(self):
        return self.size() == 0

    def __len__(self):
        return self.size()

    def __str__(self):
        return str(self.elements())

    def __eq__(self, other):
        if isinstance(other, Queue):
            return self.elements == other.elements 
        return False 




if __name__ == '__main__':
    available_backends = [list, LinkedList, DoublyLinkedList]

    for backend in available_backends:
        q1 = Queue(1,2,3,4, backend = backend)
        
        assert q1.elements() == [1,2,3,4]
        assert q1.size() == 4
        
        q1.enqueue(5)
        assert q1.elements() == [5,1,2,3,4]
        assert q1.size() == 5
        assert q1.dequeue() == 4
        assert q1.size() == 4
        assert q1.elements() == [5,1,2,3]
        assert q1.front() == 3 


        q2 = Queue(backend = backend)

        assert q2.elements() == []
        assert q2.size() == 0
        assert q2.is_empty()
        
        q2.enqueue(1)

        assert q2.elements() == [1]
        assert q2.size() == 1
        assert not q2.is_empty()
        
        if backend == LinkedList:
            print(q1.linked_list, q2.linked_list)



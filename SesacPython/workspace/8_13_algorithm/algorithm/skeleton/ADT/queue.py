import sys 
import os

# 현재 작업 디렉토리 출력
print("Current working directory:", os.getcwd())
sys.path.append('../data_structure')

try:
    from linked_list import LinkedList, LinkedNode, DoublyLinkedNode, DoublyLinkedList
except ModuleNotFoundError:
    from data_structure.linked_list import LinkedList, LinkedNode, DoublyLinkedNode, DoublyLinkedList

class Queue:
    def __init__(self, *elements, backend = list):
        assert isinstance(elements, list) or isinstance(elements, tuple)

        self.backend = backend
        if self.backend == list:
            self.list = list(elements)
        elif self.backend == LinkedList:
            self.linked_list = None # make right linked list

    def elements(self):
        if self.backend == list:
            return self.list 

    def enqueue(self, elem):
        if self.backend == list:
            self.list = [elem] + self.list

    def dequeue(self):
        if self.backend == list:
            return self.list.pop()
                
    def front(self):
        if self.backend == list:
            return self.list[-1]

    def size(self):
        if self.backend == list:
            return len(self.list)
    
    def is_empty(self):
        if self.backend == list:
            return self.list == []

    def __str__(self):
        return str(self.elements())

    def __eq__(self, other):
        if isinstance(other, Queue):
            return self.elements == other.elements 
        return False

class PriorityQueue: # 우선 순위 큐?
    def __init__(self, *elements_with_priority, backend = list):
        """
            리스트와 그 우선순위를 나타내는 2- 튜플 (OBJ,number) 목록을 가져옵니다. 숫자가 높을수록 우선 순위가 높은요소입니다..
            Get list of 2-tuple containing (obj, number), which denotes object and its priority. Higher the number, the element have hight priority. 
        """
        
    def elements(self):
        if self.backend == list:
            return self.list

    def enqueue(self, elem):
        if self.backend == list:
            self.list.append(elem)

    def dequeue(self):
        if self.backend == list:
            return self.list.pop()
                
    def front(self):
        if self.backend == list:
            return self.list[-1]

    def size(self):
        if self.backend == list:
            return len(self.list)
    
    def is_empty(self):
        if self.backend == list:
            return self.list == []

    def __str__(self):
        return str(self.elements())

    def __eq__(self, other):
        if isinstance(other, Queue):
            return self.elements == other.elements 
        return False 

if __name__ == '__main__':
    available_backends = [list, LinkedList, DoublyLinkedList]

    for backend in available_backends: # 사용가능 백앤드
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


        # q2 = Queue(backend = backend)

        # assert q2.elements() == []
        # assert q2.size() == 0
        # assert q2.is_empty()
        
        # q2.enqueue(1)

        # assert q2.elements() == [1]
        # assert q2.size() == 1
        # assert not q2.is_empty()
        
        # if backend == LinkedList:
        #     print(q1.linked_list, q2.linked_list)
    
        # q2 = PriorityQueue(('c',1), ('d',4), ('e',2), ('b',3), backend = backend)

        # assert q2.elements() == [('c',1), ('e',2), ('b',3), ('d',4)]
        # assert q2.size() == 4 
        # assert q2.front() == ('d', 4) 
        # assert not q2.is_empty()
        # q2.dequeue()

        # assert q2.elements() == [('c',1), ('d',4), ('e',2), ('b',3)]
        # assert q2.size() == 3 
        # assert q2.front() == ('b', 3) 
        # assert not q2.is_empty()

        # q2.dequeue()
        # q2.dequeue()
        # q2.dequeue()
        # q2.dequeue()

        # assert q2.is_empty()
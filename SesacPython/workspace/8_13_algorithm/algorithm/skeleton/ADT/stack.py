import sys 
sys.path.append('../data_structure')

try:
    from linked_list import LinkedList, LinkedNode, DoublyLinkedNode, DoublyLinkedList
except ModuleNotFoundError:
    from data_structure.linked_list import LinkedList, LinkedNode, DoublyLinkedNode, DoublyLinkedList

class Stack:
    def __init__(self, *elements, backend = list):
        assert isinstance(elements, list) or isinstance(elements, tuple)
        self.backend = backend
        if self.backend == list:
            self.list = list(elements)
        elif self.backend == LinkedList:
            self.linked_list = None # make right linked list 
    
    def push(self, elem):
        if self.backend == list:
            self.list.append(elem)

    def pop(self):
        if self.backend == list:
            return self.list.pop()

    def top(self):
        if self.backend == list:
            return self.list[-1]

    def is_empty(self):
        if self.backend == list:
            return self.list == []

    def size(self):
        if self.backend == list:
            return len(self.list)

if __name__ == '__main__':
    available_backends = [list, LinkedList, DoublyLinkedList]
    available_backends = [list]

    for backend in available_backends:
        s1 = Stack(3,2,1,4)
        assert s1.top() == 4 
        assert not s1.is_empty()
        assert s1.pop() == 4 
        assert s1.top() == 1 
        
        s1.push(5) 
        assert s1.top() == 5
        assert s1.size() == 4

        assert s1.pop() == 5
        assert s1.pop() == 1
        assert s1.pop() == 2
        assert s1.pop() == 3

        assert s1.is_empty()
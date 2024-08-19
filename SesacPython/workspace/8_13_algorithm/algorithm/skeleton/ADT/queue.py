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
            self.linked_list = LinkedList(elements) # make right linked list
        elif self.backend == DoublyLinkedList:
            self.doubly_linked_List = DoublyLinkedList(elements)


    def elements(self): # 모든 요소 반환
        if self.backend == list:
            return self.list
        elif self.backend == LinkedList:
            res = []
            cur = self.linked_list.head
            while cur != None:
                res.append(cur.datum)
                cur = cur.next
            return res
        elif self.backend == DoublyLinkedList:
            res = []
            cur = self.doubly_linked_List.head
            while cur != None:
                res.append(cur.datum)
                cur = cur.next
            return res

    def enqueue(self, elem):
        if self.backend == list:
            self.list = [elem] + self.list # 앞!에 요소추가
        elif self.backend == LinkedList:
            self.linked_list.append_head(elem)
        elif self.backend == DoublyLinkedList:
            self.doubly_linked_List.append_head(elem)

    def dequeue(self):
        if self.backend == list and self.list == True:
            return self.list.pop() # 마지막 요소 제거/반환
        elif self.backend == LinkedList:
            return self.linked_list.pop_end()
        elif self.backend == DoublyLinkedList:
            return self.doubly_linked_List.pop_end()
                
    def front(self):
        if self.backend == list and self.list == True: # 마지막 요소 반환
            return self.list[-1]
        elif self.backend == LinkedList:
            if self.linked_list.size != 0:
                return self.linked_list.end.datum
            else: print('There is no data to retrieve from the list.')
        elif self.backend == DoublyLinkedList:
            if self.linked_list.size != 0:
                return self.doubly_linked_List.head.datum
            else: print('There is no data to retrieve from the list.')

    def size(self):
        if self.backend == list: # size반환
            return len(self.list)
        elif self.backend == LinkedList:
            return self.linked_list.size
        elif self.backend == DoublyLinkedList:
            return self.doubly_linked_List.size
    
    def is_empty(self): 
        if self.backend == list: # 리스트가 없으면 반환
            return self.list == []
        elif self.backend == LinkedList:
            return self.linked_list.size == 0
        elif self.backend == DoublyLinkedList:
             return self.doubly_linked_List.size == 0
        return False                


    def __str__(self):  # str
        return str(self.elements())

    def __eq__(self, other):    # ==
        if isinstance(other, Queue):
            return self.elements == other.elements 
        return False

class PriorityQueue: # 우선 순위 큐
    def __init__(self, *elements_with_priority, backend = list):
        self.backend = backend
        if self.backend == list:
            self.list = list(elements_with_priority)
        elif self.backend == LinkedList:
            self.linked_list = LinkedList(elements_with_priority)
        elif self.backend == DoublyLinkedList:
            self.doubly_linked_List = DoublyLinkedList(elements_with_priority)
        """
            리스트와 그 우선순위를 나타내는 2- 튜플 (OBJ,number) 목록을 가져옵니다. 숫자가 높을수록 우선 순위가 높은요소입니다..
            Get list of 2-tuple containing (obj, number), which denotes object and its priority. Higher the number, the element have hight priority. 
        """
        assert isinstance(elements_with_priority, tuple)
        
    def elements(self):
        if self.backend == list:
            return self.list
        elif self.backend == LinkedList:
            res = []
            cur = self.linked_list.head
            while cur != None:
                res.append(cur.datum)
                cur = cur.next
            return res
        elif self.backend == DoublyLinkedList:
            res = []
            cur = self.doubly_linked_List.head
            while cur != None:
                res.append(cur.datum)
                cur = cur.next
            return res

    def enqueue(self, elem):
        if self.backend == list:
            self.list.append(elem)
            self.list.sort(key=lambda x: x[1], reverse=True)  # 요소를 뒤에 추가하고 (OBJ,number) [1]로 정렬
        elif self.backend == LinkedList:
            self.linked_list.append(elem)
            sorted_elements = sorted(self.linked_list.elements(), key=lambda x: x[1], reverse=True)
            self.linked_list = LinkedList(sorted_elements) # 요소를 뒤에 추가하고 우선순위로 정렬
        elif self.backend == DoublyLinkedList:
            pass
        
    def dequeue(self):
        if self.backend == list and self.list != 0:
            return self.list.pop()
        elif self.backend == LinkedList:
            if self.linked_list.size != 0:
                return self.linked_list.pop()
        elif self.backend == DoublyLinkedList:
            pass
                
    def front(self):
        if self.backend == list and self.list != 0:
            return self.list[-1]
        elif self.backend == LinkedList:
            if self.linked_list.size != 0:
                return self.linked_list.end.datum
            else: print('There is no data to retrieve from the list.')
        elif self.backend == DoublyLinkedList:
            if self.DoublyLinkedList.size != 0:
                return self.DoublyLinkedList.end.datum
            else: return print('There is no data to retrieve from the list.')

    def size(self):
        if self.backend == list:
            return len(self.list)
        elif self.backend == LinkedList:
            return self.linked_list.size
        elif self.backend == DoublyLinkedList:
            return self.doubly_linked_List.size
    

        elif self.backend == LinkedList:
            return self.linked_list.size == 0
        elif self.backend == DoublyLinkedList:
            return self.doubly_linked_List.size == 0
        return False
                

    def __str__(self):
        return str(self.elements())
    

    def __eq__(self, other):
        if isinstance(other, Queue):
            return self.elements == other.elements
        return False 

if __name__ == '__main__':
    available_backends = [list, LinkedList, DoublyLinkedList] # , DoublyLinkedList

    for backend in available_backends: # 사용가능 백앤드
        q1 = Queue(1,2,3,4, backend = LinkedList)
        
        assert q1.elements() == [1,2,3,4], f'{q1.elements()}'
        assert q1.size() == 4, f'{q1.size()}'
        
        q1.enqueue(5)
        assert q1.elements() == [5,1,2,3,4], f'{q1.elements()}'
        assert q1.size() == 5, f'{q1.size()}'
        assert q1.dequeue() == 4, f'{q1.dequeue()}'
        assert q1.size() == 4, f'{q1.size()}'
        assert q1.elements() == [5,1,2,3], f'{q1.elements()}'
        assert q1.front() == 3 , f'{q1.front()}'


        q2 = Queue(backend = LinkedList)

        assert q2.elements() == [], f'{q2.elements()}'
        assert q2.size() == 0, f'{q2.size()}'
        assert q2.is_empty(), f'{q2.is_empty(),q2.elements(),q2.size()}'
        
        q2.enqueue(1)

        assert q2.elements() == [1], f'{q2.elements()}'
        assert q2.size() == 1, f'{q2.size()}'
        assert not q2.is_empty(), f'{q2.is_empty(),q2.elements(),q2.size()}'
        
        if backend == LinkedList:
            print(q1.linked_list, q2.linked_list)
    
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

        # if backend == DoublyLinkedList:
        #     print(q1.doubly_linked_list, q2.doubly_linked_list)
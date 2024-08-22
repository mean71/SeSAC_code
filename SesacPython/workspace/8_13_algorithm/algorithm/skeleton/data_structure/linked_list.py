try:
    from node import Node
except ModuleNotFoundError:
    from data_structure.node import Node

class LinkedNode:
    def __init__(self, datum, next = None, node_id = None): # 식별자, 데이터 다음노드
        self.datum = datum
        self.next = next # 기본값 None
        self.node_id = node_id

# elements가 LinkedNode 클래스의 리스트
class LinkedList:

    def __init__(self, elements = None):
        self.head = None 
        self.tail = None # 어디쓸지 감이 안오니 일단 보류
        self.end = None
        self.size = 0
        self.id = 0
        if elements:
            for i in elements:
                self.append_end(i) # LinkedList.append(i)는 잘못된 방식
        elif elements is None: # 인자로 받은 리스트가 비었을때
            print('There is no data to retrieve from the list.')

        #     elements[-1].next = None
        #     self.end = LinkedNode(elements[-1])
        #     self.tail = LinkedNode(elements[1:])
        
        #     self.end = self.head
        #     self.size = 1
        #     for x,y in enumerate(elements[1:]):
        #         node = LinkedNode(x+1, y)
        #         self.end.next = node
        #         self.end = node
        #         self.size += 1

        #     assert isinstance(tail, LinkedList) or tail is None
        #     assert end.next is None
        # else:
        #     for idx, elem in enumerate(elements):
        #         if not isinstance(elem, LinkedNode):
        #             elements[idx] = LinkedNode(idx, elem)
        #     for idx, elem in enumerate(elements[:-1]):
        #         elem.next = elements[idx+1]
        #     elements[-1].next = None

        #     self.head = elements[0]
        #     self.end = elements[-1]
        #     self.tail = LinkedList(elements[1:])
        # else:
        #     size = 0
        #     for idx, elem in enumerate(elements):
        #         assert isinstance(elem, LinkedNode)
        #         if idx < len(elements) - 1:
        #             elem.next = elements[idx+1]   
        #         size += 1
            
        #     head = elements[0]
        #     tail = LinkedList(elements[1:])
        #     end = elements[-1]

            # assert isinstance(tail, LinkedList) or tail is None
            # assert end.next is None
                # node = LinkedNode(x, y)
                # self.head.next = node  # head의 .next에 node를 넣고 다시 그곳에 .next를 넣고 head.next가 같이 수정되면서 다시 .next를 생성해서 수정하고...
                # self.end = node
                # node = node.next #?
                # self.size += 1

                # node = LinkedNode(x, y)
                # self.end.next = node
                # self.end = node
                # node = node.next # 넣은 노드를 갑자기 .next로 바꿔버리면서 연결을 폭파
                # self.size += 1
                
        #연결리스트 중간에 특정 노드삽입/삭제/탐색/출력은 어떻게 구현할까? 보류...

        # 연결리스트 앞에 노드추가
    def append_head(self, elem):# 연결리스트 뒤에 노드추가
        if elem is None: return None
        self.id += 1
        self.size += 1
        new_node = LinkedNode(elem, next = None, node_id = self.id)

        if self.head:
            new_node.next = self.head
            self.head = new_node
        else:
            self.head = new_node
            self.end = new_node

    def append_end(self, elem): # 1.노드생성 2.head노드에 노드연결 3.head에 노드를 넣어 head를 바꾼다. append_end()도 동일
        if elem is None: return None
        self.id += 1
        self.size += 1
        new_node = LinkedNode(elem, next = None, node_id = self.id)

        if self.head:
            self.end.next = new_node
            self.end = new_node
        else:
            self.head = new_node
            self.end = new_node   #self.end = self.head  다르네...

        # if elem is None: return None
        # self.id += 1
        # append_head_elem = LinkedNode(self.id, elem, next = None)
        # if self.head is None:
        #     append_head_elem.next = self.head
        #     self.head = append_head_elem
        #     self.size += 1
        # else:
        #     self.id += 1
        #     self.head = LinkedNode(self.id, elem)
        #     self.size += 1

    def insert(self, idx, elem): # 연결리스트 중간에 노드추가?

        pass

    def pop_head(self): # 1. 헤드 노드 datum 꺼내서 반환 2. head.next를 head에 넣고 사이즈 1감소 3. 그럼 .next연결이 끊어진 기존 head 데이터는 어찌 처리해야 하는가?
        if self.head:
            head_data = self.head.datum
            self.head = self.head.next
            self.size -= 1
            if not self.head:
                self.end = None
            return head_data
        else: print('There is no data to retrieve from the list.')
    def pop_end(self):
        if self.head == self.end:
            end_data = self.end.datum
            self.size -= 1
            self.head = None
            self.end = None
            return end_data
        elif self.head != self.end:
            end_data = self.end.datum
            self.size -= 1
            cur = self.head
            while cur.next != self.end:
                cur = cur.next
            self.end = cur
            self.end.next = None
            return end_data
        else: print('There is no data to retrieve from the list.')

    def return_head(self):
        return self.head.datum
    def return_end(self):
        return self.end.datum
    
    def __getitem__(self, idx):
        pass
    def __setitem__(self, idx, elem):
        pass

    def __iter__(self): # 반복자로서 호출하면
        cur = self.head
        while cur is not None:
            yield cur.datum
            cur = cur.next

    def __str__(self):
        cur = self.head
        res = '[head]>> '
        while cur != None:
            res += f'[{cur.datum}] >> '
            cur = cur.next
        res += '[None]'
        return f'{res}'
        
def __str__(self):
    sur = self.head
    while cur is not None:

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
    
if __name__ =='__main__':
    # 연결리스트 테스트코드
    lst = LinkedList([1,2,3,5])
    assert lst.head.datum == 1 
    print(123)
    assert lst.head.next.datum == 2
    print(123)
    assert lst.head.next.next.datum == 3
    print(123)
    assert lst.head.next.next.next.datum == 5
    print(123)
    assert lst.head.next.next.next.next is None
    print(LinkedList([1,2,3,4]))
    print(LinkedList([1,2,3,4]).size)
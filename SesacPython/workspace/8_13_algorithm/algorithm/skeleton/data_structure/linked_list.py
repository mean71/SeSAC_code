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
class LinkedList:
    def __init__(self, elements):
        if elements == []: # 인자로 받은 리스트가 비었을때
            self.head = None
            self.end = None
            self.size = 0
        
        else:
            elements = list(elements)
            self.head = LinkedNode(0, elements[0])
            self.end = self.head
            self.size = 1
            for x,y in enumerate(elements[1:]):
                node = LinkedNode(1, 2)
                node = LinkedNode(1, 2)
                self.end.next = node
                self.end = node
                self.size += 1

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
                

    def __iter__(self): # 반복자로서 호출하면
        
        cur = self.head

        while cur is not None:
            yield cur.datum
            cur = cur.next


    def __str__(self):
        # LinkedList(리스트) 호출하면
        # res = 노드?
        node = self.head
        res = '[head]>> '
        while node != None:
            res += f'[{node.datum}] >> '
            node = node.next
        res += '[None]'
        return f'{res}'

    def _append(self, elem): # 1.노드생성 2.end노드에 노드연결 3.end에 노드를 넣어 바꾼다. appendleft()도 동일
        # if not isinstance(elem, LinkedNode):
        append_elem = LinkedNode(self.size, elem, next = None)
        self.end.next = append_elem
        self.end = append_elem
        self.size += 1

    def _pop(self):
        if self.head != None:
            head_data = self.head.datum
            self.head = self.head.next
            self.size -= 1
        else:
            print('꺼낼 데이터가 없습니다.')
        return head_data
        
        


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
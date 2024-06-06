class LinkedListNode(object):
    def __init__(self,val,next):
        self.val=val
        self.next=next


if __name__=='__main__':
    ls=LinkedListNode(1,LinkedListNode(2, LinkedListNode(3,None)))
    while ls:
        print(ls.val)
        ls=ls.next

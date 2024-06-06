class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def reverseList(self, head):
        prev = None
        current = head
        while (current.next):
            nextNode = current.next
            current.next = prev
            prev = current
            current = nextNode
        return current
if __name__=='__main__':
    print(Solution().reverseList([1,2,3,4,5]))
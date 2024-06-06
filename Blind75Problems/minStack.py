class MinStack(object):

    def __init__(self):
        self.lst = []

    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        self.lst.append(val)
        return None

    def pop(self):
        """
        :rtype: None
        """
        if len(self.lst) > 0:
            self.lst.pop(0)

    def top(self):
        """
        :rtype: int
        """
        if len(self.lst) > 0:
            return self.lst[-1]
        else:
            return 0

    def getMin(self):
        """
        :rtype: int
        """
        min_ = self.lst[0]
        return min_

obj = MinStack()
obj.push(-2)
obj.push(-2)
obj.pop()
param_3 = obj.top()
param_4 = obj.getMin()
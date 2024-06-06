class Solution(object):

    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        lst = []
        while (x != 0):
            lst.append(x % 10)
            x = x // 10
        if len(lst) <= 1:
            return True
        startIndex = 0
        endIndex = len(lst) - 1
        while (startIndex < endIndex):
            if lst[startIndex] == lst[endIndex]:
                startIndex += 1
                endIndex -= 1
            else:
                return False
        return True


Solution().isPalindrome(-121)
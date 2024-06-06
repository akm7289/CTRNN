class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        set_=set()
        for i in nums:
            if i in set_:
                return True
            else:
                set_.add(i)
        return False
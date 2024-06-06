class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        stack = []
        for i in range(0, len(nums) ):
            if len(stack) == 0 or nums[i] == stack[-1]:
                stack.append(nums[i])
            else:
                stack.pop()
        return stack[-1]

if __name__=='__main__':
    print(Solution().majorityElement([3,2,3]))
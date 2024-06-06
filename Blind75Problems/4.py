class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums == None or len(nums) < 1:
            return 0
        maxsub = nums[0]
        maxsubslidng = maxsub
        for index in range(1, len(nums)):
            maxsubslidng = maxsubslidng + nums[index]
            if maxsubslidng < nums[index]:
                maxsubslidng = nums[index]
            if maxsubslidng > maxsub:
                maxsub = maxsubslidng

        return maxsub


if __name__=="__main__":
    print(Solution().maxSubArray([-2,1]))
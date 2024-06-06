class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        min_,max_=1,1
        globalMax=max(nums)
        tmpproduct=nums[0]
        for i in range(1,len(nums)):
            tmpproduct=tmpproduct*nums[i]
            max_=max(nums[i],max_*nums[i],min_*nums[i])
            min_=min(nums[i],tmpproduct,max_*nums[i],min_*nums[i])
            globalMax=max(globalMax,max_,min_)
        return globalMax


if __name__=="__main__":
    print(Solution().maxProduct([2,3,-2,4]))
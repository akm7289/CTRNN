class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        numberLastIndex={}
        for i in range(len(nums)):
            if nums[i] in numberLastIndex:
                if k>=i-numberLastIndex[nums[i]]:
                    return True
            numberLastIndex[nums[i]]=i
        return False
if __name__=='__main__':
    print(Solution().containsNearbyDuplicate([1,2,3,1,2,3],2))
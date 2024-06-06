class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <= 1:
            return 0
        minJubms = [(0, 0)]
        index = 0
        for i in range(0, len(nums)):
            if i > minJubms[index][0]:
                index += 1

            if nums[i] + i > minJubms[index][0] and nums[i] + i>minJubms[-1][0]:
                minJubms.append((nums[i] + i, minJubms[index][1] + 1))
                if nums[i] + i >= len(nums) - 1:
                    return minJubms[index][1] + 1


if __name__=='__main__':
    print(Solution().jump([2,9,6,5,7,0,7,2,7,9,3,2,2,5,7,8,1,6,6,6,3,5,2,2,6,3]
))

class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        leftIndex = 0
        rightIndex = 0
        prev = nums[0]
        listofLists = []
        for i in range(1, len(nums)):
            if nums[i]-1 == prev:
                rightIndex += 1
            if nums[i]-1 != prev or i==len(nums)-1:
                if rightIndex > leftIndex:
                    listofLists.append(str(nums[leftIndex]) + '->' + str(nums[rightIndex]))
                    leftIndex = rightIndex = i


                else:
                    listofLists.append(str(nums[leftIndex]))
                    leftIndex+=1

            prev = nums[i]



        return listofLists




if __name__=='__main__':
    print(Solution().summaryRanges([0,1,2,4,5,7]))
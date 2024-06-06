class Solution(object):
    def maxArea(self, height):
        if height is None or len(height) < 2:
            return 0

        """
        :type height: List[int]
        :rtype: int
        """
        minPointer = 0
        maxPointer = len(height) - 1
        maxWater = (maxPointer - minPointer) * min(height[maxPointer], height[minPointer])
        while (minPointer < maxPointer):
            if height[minPointer] < height[maxPointer]:
                minPointer += 1
            else:
                maxPointer -= 1
            tmpsize = (maxPointer - minPointer) * min(height[maxPointer], height[minPointer])
            maxWater = max(maxWater, tmpsize)
        return maxWater
if __name__=="__main__":
    print(Solution().maxArea([1,8,6,2,5,4,8,3,7]))



class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """

        def startTime(val):
            return val[0]

        def inbetweenInterval(point, start, end):
            if point>=start and point<=end:
                return True
        def ThereIsOverlapping(start1,end1,start2,end2):
            if inbetweenInterval(start1,start2,end2) or inbetweenInterval(end1,start2,end2) or inbetweenInterval(start2,start1,end1) or inbetweenInterval(end2,start1,end1):
                return True
            return False

        intervals.sort(key=startTime)
        index = 1
        start, end = intervals[0]
        results = []

        while (index < len(intervals)):
            if ThereIsOverlapping(start,end,intervals[index][0],intervals[index][1]):
                start = min(start, intervals[index][0])
                end = max(end, intervals[index][1])
            else:
                results.append([start, end])
                start, end = intervals[index]
            index += 1
        results.append([start, end])
        return results


if __name__=='__main__':
    print(Solution().merge([[1,4],[2,3]]))
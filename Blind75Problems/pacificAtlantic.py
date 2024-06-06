class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        # Define a hash function to make the Point objects hashable
        return hash((self.x, self.y))

    def __eq__(self, other):
        # Define equality for Point objects
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return '[' + str(self.x) + ',' + str(self.y) + ']'







class Solution(object):
    def dfs(self,point, dataset, heights):
        i, j = point.x, point.y
        newPoint = None
        if i < 0 or j < 0 or i >= len(heights) or j >= len(heights[0]):
            return
        if i - 1 >= 0 and Point(i - 1, j) not in dataset and heights[i][j] <= heights[i - 1][j]:
            newPoint = Point(i - 1, j)
            dataset.add(newPoint)
            self.dfs(newPoint, dataset, heights)
        if i + 1 < len(heights) and Point(i + 1, j) not in dataset and heights[i][j] <= heights[i + 1][j]:
            newPoint = Point(i + 1, j)
            dataset.add(newPoint)
            self.dfs(newPoint, dataset, heights)
        if j + 1 < len(heights[0]) and Point(i, j + 1) not in dataset and heights[i][j] <= heights[i][j + 1]:
            newPoint = Point(i, j + 1)
            dataset.add(newPoint)
            self.dfs(newPoint, dataset, heights)
        if j - 1 >= 0 and Point(i, j - 1) not in dataset and heights[i][j] <= heights[i][j - 1]:
            newPoint = Point(i, j - 1)
            dataset.add(newPoint)
            self.dfs(newPoint, dataset, heights)

        return

    def pacificAtlantic(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: List[List[int]]
        """
        pacific = set()
        atlantic = set()
        for j in range(len(heights[0])):
            pacific.add(Point(0, j))
        for i in range(len(heights)):
            pacific.add(Point(i, 0))

        for j in range(len(heights[0])):
            atlantic.add(Point(len(heights) - 1, j))
        for i in range(len(heights)):
            atlantic.add(Point(i, len(heights[0]) - 1))
        pacifcopy=pacific.copy()
        for point in pacific:
            self.dfs(point,pacifcopy,heights)
        atlcopy=atlantic.copy()
        for point in atlantic:
            self.dfs(point, atlcopy,heights)


        results = pacifcopy.intersection(atlcopy)

        reslst = []
        for i in results:
            reslst.append([i.x, i.y])

        return reslst
if __name__=="__main__":
    print(Solution().pacificAtlantic([[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]))


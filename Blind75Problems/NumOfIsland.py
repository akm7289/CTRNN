class Point(object):
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
    def dfs(self, point, visitedNodes, islandx, islands):
        if point.x not in range(len(islands)) or point.y not in range(len(islands[0])):
            return

        if islands[point.x][point.y] == '0':
            visitedNodes.add(point)
            return
        else:
            x, y = point.x, point.y
            alldirections = [Point(x - 1, y), Point(x + 1, y), Point(x, y + 1), Point(x, y - 1)]
            for pointI in alldirections:
                if pointI.x in range(len(islands)) and pointI.y in range(len(islands[0])):
                    if pointI in visitedNodes:
                        continue
                    if islands[pointI.x][pointI.y] == '0':
                        visitedNodes.add(pointI)
                        continue
                    else:
                        visitedNodes.add(pointI)
                        islandx.add(pointI)
                        self.dfs(pointI, visitedNodes, islandx, islands)

    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if grid == None or len(grid) == 0:
            return 0
        visitedNodes = set()
        islands = []
        islandCount = 0
        rows, cols = len(grid), len(grid[0])
        for i in range(rows):
            for j in range(cols):
                point = Point(i, j)
                if point not in visitedNodes and grid[i][j] == '1':
                    islandCount += 1
                    islandx = set()
                    islandx.add(point)
                    visitedNodes.add(point)
                    self.dfs(point, visitedNodes, islandx, grid)
                    islands.append(islandx)
        return islandCount



if __name__=="__main__":
    grid=[["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]

    print(Solution().numIslands(grid))
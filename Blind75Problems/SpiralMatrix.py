class Solution(object):
    def spiralOrder(self, matrix):
        directions = [[0, 1], [1, 0],  [0, -1],[-1, 0]]
        currentDirction = 0
        results = []
        if not matrix or len(matrix) < 1:
            return results

        numberOfCells = len(matrix) * len(matrix[0])
        currentDirection = 0
        minrow = -1
        mincol = -1
        maxrow = len(matrix)
        maxcol = len(matrix[0])
        currenLocation = [0, -1]

        while numberOfCells > 0:
            tmp=[]
            tmp.append(currenLocation[0] + directions[currentDirction][0])
            tmp.append(currenLocation[1] + directions[currentDirction][1])
            if tmp[0] > minrow and tmp[0] < maxrow and tmp[1] > mincol and tmp[1] < maxcol:
                currenLocation = tmp
                results.append(matrix[currenLocation[0]][currenLocation[1]])
                numberOfCells -= 1
            else:
                if currentDirction == 0:
                    minrow += 1
                elif currentDirction == 1:
                    maxcol -= 1
                elif currentDirction == 2:
                    maxrow -= 1
                elif currentDirction == 3:
                    mincol += 1
                currentDirction = (currentDirction + 1) % len(directions)
        return results


if __name__=="__main__":
    print(Solution().spiralOrder([[1,2,3],[4,5,6],[7,8,9]]))
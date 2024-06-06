def solution1(S):
    if not S or len(S)==0 or not S.isnumeric():
        return 0
    lst = [0] * 10
    for i in S:
        lst[int(i)] = lst[int(i)] + 1

    result = []
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] >= 2:
            if i == 0 and len(result) == 0:
                continue
            repetition = lst[i] // 2
            tmp = [str(i)] * repetition
            result.extend(tmp)
            lst[i] -= repetition * 2
    middlechar = ''
    if sum(lst) > 0:
        for i in range(len(lst) - 1, -1, -1):
            if lst[i] != 0:
                middlechar = str(i)
                break
    left = ''
    for i in range(len(result)):
        left = left + result[i]
    return left + middlechar + left[::-1]

def DFS(plan,i,j,visitedNode):
    node=(str(i)+','+str(j))
    if i >=len(plan) or i<0 or j>=len(plan[i]) or j<0 or plan[i][j]==1 or  node in visitedNode:
        return
    visitedNode.add(node)
    if plan[i][j] == -1:
        plan[i][j] = 0
    DFS(plan,i+1,j,visitedNode)
    DFS(plan,i,j+1,visitedNode)
    DFS(plan, i - 1, j,visitedNode)
    DFS(plan, i, j - 1,visitedNode)






def convertToNum(param):
    l=[]
    for i in param:
        if i=='*':
            l.append(0)
        elif i=='#':
            l.append(1)
        elif i=='.':
            l.append(-1)
    return l
import numpy as np


def solution(plan):
    # Implement your solution here
    if not plan or len(plan)==0:
        return 0
    rows=len(plan)
    cols=len(plan[0])

    #
    lst = [[0] *(cols)]*(rows)
    lst=[]
    num_of_plans=0
    visitedCell=set()
    for i in range(rows):
        lst.append(convertToNum(plan[i]))
    for i in range(rows):
        while( -1 in lst[i]):
            num_of_plans+=1
            DFS(lst,i,lst[i].index(-1),visitedCell)

    return num_of_plans

if __name__=="__main__":
    # print(maxProfit( [1, 2, 3, 4, 5] ))
    # print(maxProfit( [23171, 21011, 21123, 21366, 21013, 21367]))
    print(solution( ['.*#..*', '.*#*.#', '######', '.*..#.', '...###']))
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def binarySearch(A,target_value,start_index,end_index):
    mid=(start_index+end_index)//2
    if start_index > end_index:
        return 0
    if A[mid]<=target_value:
        return 1
    else:
        if target_value>A[mid]:
            start_index = mid+1
        elif target_value<A[mid]:
            end_index = mid-1
        return binarySearch(A,target_value,start_index,end_index)


def solution(A):
    # Implement your solution here
    A.sort()
    num_of_triangle = 0
    for i in range(0, len(A) - 2, 1):
        sum_ = A[i] + A[i + 1]
        for j in range(i + 2, len(A)):
            if sum_ > A[j]:
                return 1
            else:
                break

    return 0

def solutionx(A):
    # Implement your solution here
    counter=0
    for i in range(len(A)-1):
        for j in range(i,len(A)):
            if i==j:
                continue
            if j-i<=A[j]+A[i]:
                counter+=1
    return counter


def solutionf(A):
    # Implement your solution here
    B = []
    counter = 0
    for index, value in enumerate(A):
        B.append(( index-value, value + index))
    B.sort(key=lambda a: a[0])
    for i in range(len(B) - 1):
        end = B[i][1]
        start = B[i][0]
        startIndex=i+1
        endIndex=len(A)
        max_counter=0
        while(startIndex<=endIndex):
            mid=(startIndex+endIndex)//2
            if mid>=len(A):
                break
            if B[mid][0] <= end :
                max_counter=max(max_counter,mid-i)
                startIndex=mid+1
            else:
                endIndex=mid-1
        counter = counter + max_counter
        if counter>10000000:
            return -1

    return counter


# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
def solution(A, B):
    # Implement your solution here
    stack = []
    for i in range(len(A)):
        if len(stack) == 0:
            stack.append((A[i], B[i]))
        else:
            if B[i] == 1:
                stack.append((A[i], B[i]))
            else:
                while (True):
                    size, direction = stack.pop()
                    if direction == B[i]:
                        stack.append((size, direction))
                        stack.append((A[i], B[i]))
                        break;
                    else:
                        if A[i] < size:
                            stack.append((size, direction))
                            break;
                        else:
                            if len(stack) == 0:
                                stack.append((A[i], B[i]))
                                break;
    return len(stack)


# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
from collections import deque

# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
from collections import deque


def getOppositeChar(i):
    if i == ")":
        return "("
    elif i == "]":
        return "["
    else:
        return "{"


def solution2(S):
    stack = deque()
    for i in S:
        if i == ")" or i == "]" or i == "}":
            char = getOppositeChar(i)

            if len(stack)==0 or char != stack.pop():
                return 0
        else:
            stack.append(i)
    if len(stack)==0:
        return 1
    else:
        return 0

def solutionss(S):
    # Implement your solution here
    stack=[]
    for i in S:
        if i==')':
            if len(stack)==0 or stack.pop()!='(':
                return 0
        else:
            stack.append(i)
    return 1 if len(stack)==0 else 0




def solutionH(H):
    # Implement your solution here
    stack=[]
    for i in H:
        if len(stack)==0:
            stack.append(i)
        else:
            for j in range(len(stack)-1,-1,-1):
                if i<stack[j]:
                    if j==0:
                        stack.append(i)
                        break;
                    else:
                        continue
                elif i==stack[j]:
                    break
                else:
                    stack.append(i)
                    break
    return len(stack)


def stone_wall(H):
    # Implement your solution here
    stack=[]
    num_of_blocks=0
    for i in H:
        if len(stack)==0:
            stack.append(i)
            num_of_blocks+=1
        else:
            if i>stack[-1]:
                stack.append(i)
                num_of_blocks+=1
            else:
                while len(stack)>0 and stack[-1]>i:
                    stack.pop()
                if len(stack)==0:
                    stack.append(i)
                    num_of_blocks+=1
                elif stack[-1]==i:
                    continue
    return num_of_blocks
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solutionequileader(A):
    # Implement your solution here
    stack =[]
    for i in A:
        if len(stack)==0:
            stack.append(i)
        elif len(stack)>0:
            if stack[-1]==i:
                stack.append(i)
            else:
                stack.pop()
    if len(stack)==0:
        return 0
    else:
        candidate=stack.pop()
        counters=[0]*len(A)
        counter=0
        for index,value in enumerate(A):
            if value==candidate:
                counter+=1
            counters[index] = counter
        final_result=0
        if counter>len(A)//2:
            for index,value in enumerate(counters):
                left_side_count=value
                right_side_count=counters[-1]-value
                if left_side_count>(index+1)//2 and right_side_count> (len(counters)-index-1)//2:
                    final_result+=1

        return final_result


def maxProfitx(A):
    # Implement your solution here
    profits=[0]*len(A)
    for i in range(len(A)-1):
        profits[i]=A[i+1]-A[i]

    maxsliding_local=maxsliding_local_global=0
    for j in range(len(A)):
        maxsliding_local=max(0,profits[i]+maxsliding_local)
        maxsliding_local_global=max(maxsliding_local,maxsliding_local_global)
    return maxsliding_local_global


def max_profit_recursively(A, start_index, end_index):
    if start_index >= end_index:
        return 0
    if start_index+1==end_index:
        if end_index<len(A):
            return A[end_index]-A[start_index]
        else :
            return 0
    mid = (start_index + end_index) // 2

    max_price = min_price = A[mid]

    for i in range(mid, start_index-1, -1):
        if A[i] < min_price:
            min_price = A[i]
    for j in range(mid, end_index, 1):
        if A[j] > max_price:
            max_price = A[j]
    max_profit=max_price - min_price
    left_max=max_profit_recursively(A, start_index, mid)
    right_max= max_profit_recursively(A, mid, end_index)
    return max(max_profit,left_max ,right_max)

import math
import sys
def solutionperimeter(N):
    min_perimeter=sys.maxsize
    sqrtN=int(math.sqrt(N))
    for i in range(1,sqrtN+1):
        if N%i==0:
            min_perimeter=min(min_perimeter,int(2*(i+N/i)))
    # Implement your solution here
    return min_perimeter
def maxProfit(A):
    # Implement your solution here
    start_index = 0
    end_index = len(A)

    return max_profit_recursively(A, start_index, end_index)


# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
import math
def trypeaks(i, peaks):
    counter = i
    k = 0
    iterator = 0
    k = peaks[iterator]
    counter -= 1
    iterator += 1
    while k <= peaks[-1] and iterator < len(peaks):
        if peaks[iterator] - k >= i:
            counter -= 1
            if counter<=0:
                break;
            k = peaks[iterator]
        iterator += 1

    if counter <= 0:
        return True
    return False

def solution(A):
    # Implement your solution here
    if A is None or len(A)<=2:
        return 0
    peaks=[]
    for i in range(1,len(A)-1):
        if A[i]>A[i-1] and A[i]>A[i+1]:
            peaks.append(i)
    num_of_peaks=min(len(peaks),int(math.sqrt(len(A)))+1)
    if num_of_peaks<2:
        return num_of_peaks
    for i in range(num_of_peaks,-1,-1):
        if trypeaks(i,peaks):
            return i
    return 0


def solutionff(A):
    # Implement your solution here
    peaks = []
    for i in range(1, len(A) - 1):
        if A[i] > A[i - 1] and A[i] > A[i + 1]:
            peaks.append(i)
    num_of_peaks = len(peaks)
    for i in range(num_of_peaks , -1, -1):
        if trypeaks(i, peaks):
            return i
    return 0


# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
import math


def solution(A):
    # Implement your solution here
    if len(A)<=2:
        return 0
    peaks = [0] * len(A)
    for i in range(1, len(A) - 1):
        if A[i] > A[i + 1] and A[i] > A[i - 1]:
            peaks[i] = 1
    if sum(peaks)<2:
        return 0
    num_peaks=min(int(math.sqrt(len(A)) + 1),sum(peaks))
    for j in range(int(math.sqrt(len(A)) + 1),1, -1):
        if len(A) % j == 0:
            k = 0
            shift = len(A) // j
            allHavePeaks = True
            while k < len(peaks):
                if sum(A[k:k + shift]) > 0:
                    k = k + shift
                    continue
                else:
                    allHavePeaks = False
                    break
            if allHavePeaks:
                return shift
    return len(A)


# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(M, A):
    # Implement your solution here
    intSet = set()
    result = 0
    intSet.add(A[0])
    j = 1
    for i in range(len(A)):

        if i >= 1:
            intSet.remove(A[i - 1])

        while (j < len(A)):
            if A[j] in intSet:
                break
            else:
                intSet.add(A[j])
                j += 1
        result += len(intSet)
        if result >= 1000000000:
            return 1000000000
    if result >= 1000000000:
        return 1000000000
    return result


def solutionfind(A):
    # Implement your solution here
    lst=[False]*(len(A))

    for i in A:
        if i<=len(A) and i>0:
            lst[i-1]=True
    for index,boolean in enumerate(lst):
        if not boolean:
            return index+1
    return len(A)+1
if __name__=="__main__":
    # print(maxProfit( [1, 2, 3, 4, 5] ))
    # print(maxProfit( [23171, 21011, 21123, 21366, 21013, 21367]))
    print(solutionfind( [-1, -3]))




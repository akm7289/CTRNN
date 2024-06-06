def badPairs(sequence):
    for i in range(1,len(sequence)):
        if sequence[i-1]>=sequence[i]:
            return i-1,i
def checkIncreasingSkipIndex(sequence,skipIndex):
    for i in range(0,len(sequence)-1):
        if i==skipIndex:
            if i-1>=0 and i+1<len(sequence):
                if sequence[i-1]>=sequence[i+1]:
                    return False
                else:
                    continue
            else:
                continue
        if i+1==skipIndex:
            if i+2<len(sequence):
                if sequence[i]>=sequence[i+2]:
                    return False
                else:
                    continue
            else:
                continue

        if sequence[i]>=sequence[i+1]:
            return False
    return True
def solution(sequence):
    i,j=badPairs(sequence)
    return checkIncreasingSkipIndex(sequence,i) or checkIncreasingSkipIndex(sequence,j)



print(solution([1, 2, 5, 3, 5]))
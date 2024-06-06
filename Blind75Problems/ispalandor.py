def solution(inputString):
    lettersCount = {}
    coutner = 0
    for char in inputString:
        if char in lettersCount:
            lettersCount[char] += 1
        else:
            lettersCount[char] = 1
        coutner += 1
    oneOddAllowed = True
    if coutner % 2 == 0:
        oneOddAllowed = False

    for key, value in lettersCount.items():
        if value % 2 == 0:
            if oneOddAllowed:
                oneOddAllowed = False
            else:
                return False

    return True

print(solution('aabb'))
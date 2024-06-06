def decode(file_path):
    file_text = open(file_path, "r")# reading the file as string
    lines = file_text.readlines()
    dictionary = {}  # initilize empty map
    lst = []
    for line in lines:
        key, word=line.split(' ')
        dictionary[key]=word
        lst.append(key)
    lst.sort()
    jump = 1
    index = 0
    output = ''
    while (index < len(lst)):
        output=output+str(lst[index])+':'+dictionary[lst[index]]+'\n'
        jump=jump+1  # to jumb from 1 to 3 to 6...
        index=index+jump

    return output

print(decode('coding_qual_input.txt'))
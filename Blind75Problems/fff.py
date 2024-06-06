def solution(queries):
    map = {}
    results = []
    for query in queries:
        if query[0] == 'TOP_N_KEYS':
            sortedLst = sorted(map.items(), key=lambda x: (x[1]['count'], x[0]), reverse=True)

            sorted_keys_and_count = [(k, v['count']) for k, v in sortedLst]
            rst = ''
            index = min(int(query[1]),len(sorted_keys_and_count))
            for i in range(index):
                rst += sorted_keys_and_count[i][0] + '(' + str(sorted_keys_and_count[i][1]) + ')'
            results.append(rst)
            continue

        if query[1] in map:
            if query[0] == 'SET_OR_INC':
                if query[2] in map[query[1]]:
                    map[query[1]][query[2]] = str(int(map[query[1]][query[2]]) + int(query[3]))
                    results.append(str(map[query[1]][query[2]]))
                else:
                    map[query[1]][query[2]] = str(query[3])
                    results.append(str(query[3]))
                map[query[1]]['count'] = map[query[1]]['count'] + 1
            elif query[0] == 'GET':
                if query[2] in map[query[1]]:
                    results.append(str(map[query[1]][query[2]]))
                else:
                    results.append("")

            elif query[0] == 'DELETE':
                if query[2] in map[query[1]]:
                    results.append("true")
                    del map[query[1]][query[2]]
                    map[query[1]]['count'] = map[query[1]]['count'] + 1
                else:
                    results.append("false")
        else:
            if query[0] == 'SET_OR_INC':
                map[query[1]] = {}
                map[query[1]][query[2]] = str(query[3])
                map[query[1]]['count'] = 1
                results.append(str(query[3]))
            elif query[0] == 'GET':
                results.append("")
            elif query[0] == 'DELETE':
                results.append("false")

    return results


print(solution(
[["SET_OR_INC","John","experience","10"],
 ["SET_OR_INC","John","age","36"],
 ["SET_OR_INC","age","Jhon","2"],
 ["GET","Jhon","experience"],
 ["TOP_N_KEYS","1"],
 ["SET_OR_INC","James","height","160"],
 ["SET_OR_INC","James","age","30"],
 ["SET_OR_INC","James","age","1"],
 ["TOP_N_KEYS","2"],
 ["SET_OR_INC","John","height","175"],
 ["TOP_N_KEYS","2"],
 ["TOP_N_KEYS","1"],
 ["SET_OR_INC","James","experience","5"],
 ["SET_OR_INC","James","experience","7"],
 ["TOP_N_KEYS","2"]]))
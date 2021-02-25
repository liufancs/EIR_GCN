user_items, item_users = {}, {}
with open('train.txt', 'r') as file:
    for f in file.readlines():
        temp = f.rstrip('\n').split(' ')
        user = temp[0]
        items = temp[1:]
        user_items[user] = items
        for item in items:
            if item not in item_users.keys():
                item_users[item] = []
            item_users[item].append(user)

train_uu = []
sorted_pict_len_u = 100
for user in user_items.keys():
    user_list = []
    temp_items = user_items[user]
    for item in temp_items:
        user_list+=item_users[item]
    countDict = dict()
    pDict = dict()
    user_drop = set(user_list)
    for i in user_drop:
        # countDict[i] = user_list.count(i)
        pDict[i] = user_list.count(i)/len(user_list)
    #new_user_list = [k for k,v in pDict.items() if (v > 0.005 and k != user)]
    del pDict[user]
    sorted_pict = (sorted(pDict.items(), key=lambda item:item[1], reverse=True))
    new_user_list = []
    #sorted_pict = sorted_pict[:int(0.1 * len(sorted_pict))]
    sorted_pict_len_u = len(sorted_pict)
    if sorted_pict_len_u > 50:
        sorted_pict = sorted_pict[:50]
    else:
        print('not enough')

    for k, v in sorted_pict:
        new_user_list.append(k)
    if new_user_list == []:
        pass
    else:
        train_uu.append([user] + new_user_list)

train_ii = []
sorted_pict_len = 100
for item in item_users.keys():
    item_list = []
    temp_users = item_users[item]
    for user in temp_users:
        item_list+=user_items[user]
    countDict = dict()
    pDict = dict()
    item_drop = set(item_list)
    for i in set(item_list):
        pDict[i] = item_list.count(i) / len(item_list)
    #new_item_list = [k for k,v in pDict.items() if (v > 0.005 and k != item)]
    del pDict[item]
    sorted_pict = (sorted(pDict.items(), key=lambda item: item[1], reverse=True))
    new_item_list = []
    #sorted_pict = sorted_pict[:int(0.1*len(sorted_pict))]
    sorted_pict_len = len(sorted_pict)
    if sorted_pict_len > 50:
        sorted_pict = sorted_pict[:50]
    else:
        print('not enough')

    for k, v in sorted_pict:
        new_item_list.append(k)
    if new_item_list == []:
        pass
    else:
        train_ii.append(([item] + new_item_list))

with open('train_u_t50.txt', 'w') as file:
    for i in range(len(train_uu)):
        sorted_pict_len_u = len(train_uu[i])
        temp = ''
        if sorted_pict_len_u < 50:
            for j in range(sorted_pict_len_u):
                if j == sorted_pict_len_u - 1:
                    temp += train_uu[i][j] + '\n'
                else:
                    temp += train_uu[i][j] + ' '
        else:
            for j in range(50):
                if j == 50 - 1:
                    temp += train_uu[i][j] + '\n'
                else:
                    temp += train_uu[i][j] + ' '
        file.writelines(temp)

with open('train_u_t40.txt', 'w') as file:
    for i in range(len(train_uu)):
        sorted_pict_len_u = len(train_uu[i])
        temp = ''
        if sorted_pict_len_u < 40:
            for j in range(sorted_pict_len_u):
                if j == sorted_pict_len_u - 1:
                    temp += train_uu[i][j] + '\n'
                else:
                    temp += train_uu[i][j] + ' '
        else:
            for j in range(40):
                if j == 40 - 1:
                    temp += train_uu[i][j] + '\n'
                else:
                    temp += train_uu[i][j] + ' '
        file.writelines(temp)

with open('train_u_t30.txt', 'w') as file:
    for i in range(len(train_uu)):
        sorted_pict_len_u = len(train_uu[i])
        temp = ''
        if sorted_pict_len_u < 30:
            for j in range(sorted_pict_len_u):
                if j == sorted_pict_len_u - 1:
                    temp += train_uu[i][j] + '\n'
                else:
                    temp += train_uu[i][j] + ' '
        else:
            for j in range(30):
                if j == 30 - 1:
                    temp += train_uu[i][j] + '\n'
                else:
                    temp += train_uu[i][j] + ' '
        file.writelines(temp)

with open('train_u_t20.txt', 'w') as file:
    for i in range(len(train_uu)):
        sorted_pict_len_u = len(train_uu[i])
        temp = ''
        if sorted_pict_len_u < 20:
            for j in range(sorted_pict_len_u):
                if j == sorted_pict_len_u - 1:
                    temp += train_uu[i][j] + '\n'
                else:
                    temp += train_uu[i][j] + ' '
        else:
            for j in range(20):
                if j == 20 - 1:
                    temp += train_uu[i][j] + '\n'
                else:
                    temp += train_uu[i][j] + ' '
        file.writelines(temp)

with open('train_u_t10.txt', 'w') as file:
    for i in range(len(train_uu)):
        sorted_pict_len_u = len(train_uu[i])
        temp = ''
        if sorted_pict_len_u < 10:
            for j in range(sorted_pict_len_u):
                if j == sorted_pict_len_u - 1:
                    temp += train_uu[i][j] + '\n'
                else:
                    temp += train_uu[i][j] + ' '
        else:
            for j in range(10):
                if j == 10 - 1:
                    temp += train_uu[i][j] + '\n'
                else:
                    temp += train_uu[i][j] + ' '
        file.writelines(temp)

with open('train_i_t50.txt', 'w') as file:
    for i in range(len(train_ii)):
        sorted_pict_len_i = len(train_ii[i])
        temp = ''
        if sorted_pict_len_i < 50:
            for j in range(sorted_pict_len_i):
                if j == sorted_pict_len_i - 1:
                    temp += train_ii[i][j] + '\n'
                else:
                    temp += train_ii[i][j] + ' '
        else:
            for j in range(50):
                if j == 50 - 1:
                    temp += train_ii[i][j] + '\n'
                else:
                    temp += train_ii[i][j] + ' '
        file.writelines(temp)

with open('train_i_t40.txt', 'w') as file:
    for i in range(len(train_ii)):
        sorted_pict_len_i = len(train_ii[i])
        temp = ''
        if sorted_pict_len_i < 40:
            for j in range(sorted_pict_len_i):
                if j == sorted_pict_len_i - 1:
                    temp += train_ii[i][j] + '\n'
                else:
                    temp += train_ii[i][j] + ' '
        else:
            for j in range(40):
                if j == 40 - 1:
                    temp += train_ii[i][j] + '\n'
                else:
                    temp += train_ii[i][j] + ' '
        file.writelines(temp)

with open('train_i_t30.txt', 'w') as file:
    for i in range(len(train_ii)):
        sorted_pict_len_i = len(train_ii[i])
        temp = ''
        if sorted_pict_len_i < 30:
            for j in range(sorted_pict_len_i):
                if j == sorted_pict_len_i - 1:
                    temp += train_ii[i][j] + '\n'
                else:
                    temp += train_ii[i][j] + ' '
        else:
            for j in range(30):
                if j == 30 - 1:
                    temp += train_ii[i][j] + '\n'
                else:
                    temp += train_ii[i][j] + ' '
        file.writelines(temp)

with open('train_i_t20.txt', 'w') as file:
    for i in range(len(train_ii)):
        sorted_pict_len_i = len(train_ii[i])
        temp = ''
        if sorted_pict_len_i < 20:
            for j in range(sorted_pict_len_i):
                if j == sorted_pict_len_i - 1:
                    temp += train_ii[i][j] + '\n'
                else:
                    temp += train_ii[i][j] + ' '
        else:
            for j in range(20):
                if j == 20 - 1:
                    temp += train_ii[i][j] + '\n'
                else:
                    temp += train_ii[i][j] + ' '
        file.writelines(temp)

with open('train_i_t10.txt', 'w') as file:
    for i in range(len(train_ii)):
        sorted_pict_len_i = len(train_ii[i])
        temp = ''
        if sorted_pict_len_i < 10:
            for j in range(sorted_pict_len_i):
                if j == sorted_pict_len_i - 1:
                    temp += train_ii[i][j] + '\n'
                else:
                    temp += train_ii[i][j] + ' '
        else:
            for j in range(10):
                if j == 10 - 1:
                    temp += train_ii[i][j] + '\n'
                else:
                    temp += train_ii[i][j] + ' '
        file.writelines(temp)
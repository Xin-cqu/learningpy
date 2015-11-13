wine_list = [7, 0, 0, 0]


def drink(wine_list):
    wine_list[3] += wine_list[0]
    wine_list[1] += wine_list[0]
    wine_list[2] += wine_list[0]
    wine_list[0] = 0

    return wine_list


def exchange(wine_list):
    wine_list[0] = int(wine_list[1] / 2) + int(wine_list[2] / 4)
    wine_list[1] = wine_list[1] % 2
    wine_list[2] = wine_list[2] % 4
    return wine_list


loop = 0
while wine_list[0] > 0:
    drink(wine_list)
    exchange(wine_list)
    loop += 1
print("bottles", wine_list[1])
print("shell", wine_list[2])
print("exchange time", loop)
print("total drink", wine_list[3])

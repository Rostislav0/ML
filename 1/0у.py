v = [int(i) for i in input().split()]
if len(v) < 2 or len(v) % 2 == 0:
    print('Ошибка. Кучек слишком мало, чтобы можно было решить задачу.')
elif sum(v) % 2 == 0:
    print('Кучки можно уравнять')
else:
    print('Кучки нельзя уравнять')

cinema = {}
for i in range(1,11):
    print(f'Cate filme sunt in sala {i} ?')
    cinema[f'Sala {i}'] = [[],[],[]]
    while True:
        try:
            nr_filme = int(input())
            break
        except ValueError:
            print(f'Trebuie sa introduci un numar!!!')
    cinema[f'Sala {i}'][0].append(nr_filme)
    for j in range(1,nr_filme+1):
        film = input(f'Film {j}: ')
        cinema[f'Sala {i}'][1].append(film)
        ora = input('Ora: ')
        ora = ora.split()
        cinema[f'Sala {i}'][2].append(ora)
# print(cinema)
print()
for i in range(1,11):
    print(f'<hall{i}>')
    for j in range(cinema[f'Sala {i}'][0][0]):
        print(f"<a title='{cinema[f'Sala {i}'][1][j]}' h='{cinema[f'Sala {i}'][2][j][0]}' m='{cinema[f'Sala {i}'][2][j][1]}' />")
    print(f'</hall{i}>')


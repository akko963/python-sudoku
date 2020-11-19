# Sudoku Game - Sudoku Checker
# 1. Sudoku game is a class that takes an initial array with existing elements that form initial puzzle.
#        It consists of 2-D array [0:9][0:9] initialized to 0 at empty spots.
# 2. It has a randomly created sudoku maker which initialize the sudoku with randomly initialized elements
# 3. It has a checker to check whether a sudoku with added input is valid
from itertools import product
from random import choice, choices, randint

def generate_lookup():
    keys, values = [], [[] for i in range(4)]
    for i in range(9):
        for j in range(9):
            keys.append((i, j))  # first 81 keys
            values[0].append((i, j))  # first 81 rows
            values[1].append((j, i))  # first 81 cols
    for i in range(9):
        for j in range(9):  # 2*81 key: row
            keys.append(('rows', i, j))
            values[0].append((i, j))  # 160
            values[1].append((j, i))  # 160
    for i in range(9):  # 3*81 key: col
        for j in range(9):
            keys.append(('cols', j, i))
            values[0].append((i, j))  # 270
            values[1].append((j, i))  # 270
    for i in range(81):  # block first 81
        x, y = keys[i]
        p = (x // 3) * 3 + (y // 3)  # block-number
        q = (x % 3) * 3 + (y % 3)  # block-index
        values[2].append((p, q))  # block 81
    for i in range(81):
        values[2].append(values[2][i])  # block 80,160
    for i in range(81):
        x, y = values[2][i]
        keys.append(('blocks', x, y))
        values[0].append(values[0][i])
        values[1].append(values[1][i])
    for i in range(81):
        values[2].append(values[2][i])  # block 160, 270 (col)  # same lookup tables 0-81  since col 0-81 is block 0-81
    for i in range(81):
        values[2].append(values[2][i])  # block 270, 320 (block)
    pair = "cols", "blocks"
    for k in range(81 * 4):
        if (2 * 81) <= k < (3 * 81):
            pair = "rows", "blocks"
        elif (3 * 81) <= k < (4 * 81):
            pair = "rows", "cols"
        look = {'cols': 1, 'rows': 0, 'blocks': 2}
        values[3].append(((pair[0], values[look[pair[0]]][k][0]), (pair[1], values[look[pair[1]]][k][0])))
    lookup = {}
    test = list(zip(values[0], values[1], values[2], values[3]))
    for k in range(81 * 4):
        lookup[keys[k]] = test[k]
    return lookup


def show_args(func):
    def _wrapper(*args, **kwargs):
        print("Test func arguments: ", len(args), "args", len(kwargs), "kwargs")
        print("Args:", *args, **kwargs)
        return func(*args, **kwargs)

    return func


def printList(lst, msg=None, levels=None):
    print()
    if msg:
        print(msg)
    ch = "\n" if msg is None or msg != 0 else ""
    if not levels:
        print(ch.join(f'{item}' for item in lst))


def print_sudoku(slots):
    for i, s in enumerate(slots):
        print("".join([f'{s[j]} ' for j in range(9)]))
    print()


def collision_entries(slots):
    slots = sorted(set(slots))
    if len(slots) <= 1:
        return slots
    for i in range(len(slots) - 1):
        if slots[i][0] == slots[i + 1][0] and slots[i][1] == slots[i + 1][1]:
            return []
    return slots


def translate(source, x, y, to="rows"):
    if source == to:
        return x, y
    return ref[(source, x, y)][0]


def translate_list(source, x, y, to="rows"):
    if source == to:
        return [(x, j) for j in y]
    if source == "rows" or source == "cols":
        return [(j, x) for j in y]
    return [ref[(source, x, k)][0] for k in y]


def converter(func):
    def _wrap(*args, **kwargs) -> list:
        idx = func(*args, **kwargs)
        converted = [[0 for i in range(9)] for j in range(9)]
        for i in range(9):
            for j in range(9):
                p, q = ref[(i, j)][idx]
                converted[p][q] = args[0][i][j]
        return converted

    return _wrap


@converter
def convert_to_block(slots=None): return 2


@converter
def convert_to_original(slots=None): return 2


@converter
def convert_rotate(slots=None): return 1


def has_no_dupes(slots=None):
    for new_list in slots:
        count = sum(x > 0 for x in new_list)
        set_l = set(new_list) - {0}
        if count != len(set_l):
            return False
    return True


def check_sudoku(s):
    my_sudoku, cols, blocks = s, [[s[j][i] for i in range(9)] for j in range(9)], convert_to_block(s)
    return has_no_dupes([*my_sudoku, *cols, *blocks])


def check_progress(slots):
    total = 0
    for i in range(9):
        total += sum(x > 0 for x in slots[i])
    return total


def make_list(lst=None):
    return list([0 for i in range(9)] for j in range(9))


play_set = set(range(1, 10))    # {1, 2, 3, 4, 5, 6, 7, 8, 9}
all_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
slot_set = {0, 1, 2, 3, 4, 5, 6, 7, 8}
ref = generate_lookup()
table = {0: 'rows', 1: 'cols', 2: 'blocks'}
table2 = {'rows': 0, 'cols': 1, 'blocks': 2}

board = [[1 + i + 9 * j for i in range(9)] for j in range(9)]
board2 = make_list()
for i in range(9):
    for j in range(9):
        p, q = ref[(i, j)][table2['blocks']]  # change the original indices: i, j to blocks: p, q
        board2[p][q] = board[i][j]

def test_block_conversion():
    test_board = make_list()
    for i in range(9):
        for j in range(9):
            p, q = ref[(i, j)][table2['blocks']]  # change the original indices: i, j to blocks: p, q
            test_board[p][q] = board[i][j]
    assert convert_to_block(board) == test_board


def test_col_conversion():
    test_board = make_list()
    for i in range(9):
        for j in range(9):
            p, q = ref[(i, j)][table2['cols']]  # change the original indices: i, j to blocks: p, q
            test_board[p][q] = board[i][j]
    assert convert_rotate(board) == test_board


def test_col_lookups():
    row_list, col_list = [], []
    for i in range(9):
        for j in range(9):
            row_list.append((i, j))
            col_list.append((j, i))
    for col in col_list:
        assert col == ref[col][table2['rows']]
    for row in row_list:
        i, j = row
        assert (j, i) == ref[row][table2['cols']]


def test_block_lookups():
    row_list = []
    for i in range(9):
        for j in range(9):
            row_list.append((i, j))
    for row in row_list:
        i, j = row
        lookup_i, lookup_j = ref[row][table2['blocks']]
        assert board[lookup_i][lookup_j] == board2[i][j]

def test_generate_lookup():
    assert len(ref.keys()) == 81 * 4
    my_list = list(product([i for i in range(9)], repeat=2))
    for i in range(81):
        assert len(set(my_list)) == 81
        assert 0 <= my_list[i][0] <= 9 and 0 <= my_list[i][1] <= 9
        assert ref[my_list[i]]
        assert ref[('rows', *my_list[i])]
        assert ref[('cols', *my_list[i])]
        assert ref[('blocks', *my_list[i])]

        assert 0 <= ref[my_list[i]][0][0] <= 9 and 0 <= ref[my_list[i]][0][1] <= 9
        assert 0 <= ref[my_list[i]][1][0] <= 9 and 0 <= ref[my_list[i]][1][1] <= 9
        assert 0 <= ref[my_list[i]][2][0] <= 9 and 0 <= ref[my_list[i]][2][1] <= 9

        assert 0 <= ref[('rows', *my_list[i])][0][0] <= 9 and 0 <= ref[('rows', *my_list[i])][0][1] <= 9
        assert 0 <= ref[('rows', *my_list[i])][1][0] <= 9 and 0 <= ref[('rows', *my_list[i])][1][1] <= 9
        assert 0 <= ref[('rows', *my_list[i])][2][0] <= 9 and 0 <= ref[('rows', *my_list[i])][2][1] <= 9

        assert 0 <= ref[('cols', *my_list[i])][0][0] <= 9 and 0 <= ref[('cols', *my_list[i])][0][1] <= 9
        assert 0 <= ref[('cols', *my_list[i])][1][0] <= 9 and 0 <= ref[('cols', *my_list[i])][1][1] <= 9
        assert 0 <= ref[('cols', *my_list[i])][2][0] <= 9 and 0 <= ref[('cols', *my_list[i])][2][1] <= 9

        assert 0 <= ref[('blocks', *my_list[i])][0][0] <= 9 and 0 <= ref[('blocks', *my_list[i])][0][1] <= 9
        assert 0 <= ref[('blocks', *my_list[i])][1][0] <= 9 and 0 <= ref[('blocks', *my_list[i])][1][1] <= 9
        assert 0 <= ref[('blocks', *my_list[i])][2][0] <= 9 and 0 <= ref[('blocks', *my_list[i])][2][1] <= 9

        assert ref[my_list[i]][3][0][0] in ['rows', "cols", "blocks"] and ref[my_list[i]][3][1][0] in ['rows', "cols",
                                                                                                       "blocks"]
        assert 0 <= ref[my_list[i]][3][0][1] <= 9 and 0 <= ref[my_list[i]][3][1][1] <= 9

        assert ref[('rows', *my_list[i])][3][0][0] in ['rows', "cols", "blocks"] and ref[('rows', *my_list[i])][3][1][
            0] in ['rows', "cols", "blocks"]
        assert 0 <= ref[('rows', *my_list[i])][3][0][1] <= 9 and 0 <= ref[('rows', *my_list[i])][3][1][1] <= 9

        assert ref[('cols', *my_list[i])][3][0][0] in ['rows', "cols", "blocks"] and ref[('cols', *my_list[i])][3][1][
            0] in ['rows', "cols", "blocks"]
        assert 0 <= ref[('cols', *my_list[i])][3][0][1] <= 9 and 0 <= ref[('cols', *my_list[i])][3][1][1] <= 9

        assert ref[('blocks', *my_list[i])][3][0][0] in ['rows', "cols", "blocks"] and \
               ref[('blocks', *my_list[i])][3][1][0] in ['rows', "cols", "blocks"]
        assert 0 <= ref[('blocks', *my_list[i])][3][0][1] <= 9 and 0 <= ref[('blocks', *my_list[i])][3][1][1] <= 9

        # assert ref[('rows',*my_list[i])]
        # assert ref[('cols', *my_list[i])]
        # assert ref[('blocks',*my_list[i])]


def test_has_no_dupes():
    assert has_no_dupes(board)


def test_check_sudoku():
    assert check_sudoku(board)


def test_convert_to_block():
    assert convert_to_block(board) == board2


def test_convert_to_original():
    assert convert_to_block(board2) == board


def test_translate_list():
    def r1() : return choice(['rows', 'cols', 'blocks'])
    def r2() : return randint(0,8)
    for i in range(1000):
        y_list = choices(range(0,9),k=randint(2,30))
        source, index = r1(), r2()
        ans = translate_list(source, index, y_list)
        assert len(y_list) == len(ans)
        for k in ans:
            assert 0 <= k[0] <= 9 and 0 <= k[1] <= 9
            for i, y in enumerate(y_list):
                if source == "rows":
                    assert (index, y) == ans[i]
                if source == "cols":
                    assert (y, index) == ans[i]
                if source == "blocks":
                    assert board2[index][y] == board[ans[i][0]][ans[i][1]]
    assert True

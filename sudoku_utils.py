# Sudoku Game - Sudoku Checker
# 1. Sudoku game is a class that takes an initial array with existing elements that form initial puzzle.
#        It consists of 2-D array [0:9][0:9] initialized to 0 at empty spots.
# 2. It has a randomly created sudoku maker which initialize the sudoku with randomly initialized elements
# 3. It has a checker to check whether a sudoku with added input is valid
from itertools import product
from random import choice, choices, randint

ROWS = 9
SIZE = ROWS*ROWS

def generate_lookup():
    row = "rows"
    col = "cols"
    block = "blocks"
    keys, values = [], [[] for i in range(4)]

    keys = [(i, j) for i in range(ROWS) for j in range(ROWS)]
    values[0] = [(i, j) for i in range(ROWS) for j in range(ROWS)]
    values[1] = [(j, i) for i in range(ROWS) for j in range(ROWS)]

    for i in range(SIZE):  # first SIZE blocks   #values #2
        x, y = keys[i]
        _p = (x // 3) * 3 + (y // 3)  # block-number
        _q = (x % 3) * 3 + (y % 3)  # block-index
        values[2].append((_p, _q))  # block SIZE

    # Fill the rest by copying the above horizontal row
    # SIZE (to) 2*SIZE
    keys += [(row, *values[0][i]) for i in range(SIZE)]
    keys += [(col, *values[1][i]) for i in range(SIZE)]
    keys += [(block, *values[2][i]) for i in range(SIZE)]
    # SIZE (to) 2*SIZE fill the rest by copying the ones from the horizontal rows above
    values[0] += values[0][0:SIZE]
    values[1] += values[1][0:SIZE]
    values[2] += values[2][0:SIZE]
    # 2*SIZE (to) 3*SIZE
    values[0] += values[1][0:SIZE]
    values[1] += values[0][0:SIZE]
    values[2] += values[2][0:SIZE]
    # 3*SIZE (to) 4*SIZE
    values[0] += values[0][0:SIZE]
    values[1] += values[1][0:SIZE]
    values[2] += values[2][0:SIZE]

    _xy, _xy1, _xy2 = values[0][0:SIZE], values[1][0:SIZE], values[2][0:SIZE]
    _r = [(i, j) for i, j in _xy]
    _c, _b = [(i + ROWS, j) for i, j in _xy1], \
             [(i + ROWS*2, j) for i, j in _xy2],
    _p_keys = [*_r, *_c, *_b]
    _p_values = [[*_r, *_r, *_r],
                 [*_c, *_c, *_c],
                 [*_b, *_b, *_b], [], [], []]

    # generate others
    ref_a, ref_b = 1, 2
    for idx, _ in enumerate(_p_keys[0:SIZE*3]):
        if SIZE <= idx < 2*SIZE:
            ref_a, ref_b = 0, 2
        elif 2*SIZE <= idx < 3*SIZE:
            ref_a, ref_b = 0, 1
        _p_values[3].append((_p_values[ref_a][idx], _p_values[ref_b][idx]))
        _p_values[4].append((_p_values[ref_a][idx][0], _p_values[ref_b][idx][0]))
    _p_values[5] = [*_xy, *_xy, *_xy]

    REFP = dict()
    for idx, _p_value in enumerate(zip(*_p_values)):
        REFP[_p_keys[idx]] = _p_value

    pair = "cols", "blocks"
    for k in range(SIZE * 4):
        if (2 * SIZE) <= k < (3 * SIZE):
            pair = "rows", "blocks"
        elif (3 * SIZE) <= k < (4 * SIZE):
            pair = "rows", "cols"
        look = {'cols': 1, 'rows': 0, 'blocks': 2}
        values[3].append(((pair[0], values[look[pair[0]]][k][0]), (pair[1], values[look[pair[1]]][k][0])))

    lookup = dict()
    test = list(zip(*values))
    for k in range(SIZE * 4):
        lookup[keys[k]] = test[k]
    return lookup, REFP, values

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
        print("".join([f'{s[j]} ' for j in range(ROWS)]))
    print()

def freq(my_list: list, reverse: bool=True):
    """ :returns list (play, freq) )"""
    pot = my_list
    if not my_list:
        return []
    if type(my_list[0]) == list:
        pot = [play for g in my_list for play in g]
    elif type(my_list[0]) == set:
        pot = [play for _set in my_list for play in list(_set)]
    plays = list(set(pot))
    tb = [0]*len(plays)
    for _p in pot:
        tb[plays.index(_p)] += 1
    return sorted(zip(plays, tb), key=lambda _k: _k[1], reverse=reverse)

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
        converted = [[0 for i in range(ROWS)] for j in range(ROWS)]
        for i in range(ROWS):
            for j in range(ROWS):
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

def has_no_dupes(slots):
    for new_list in slots:
        count = sum(x > 0 for x in new_list)
        set_l = set(new_list) - {0}
        if count != len(set_l):
            return False
    return True

def check_sudoku(s):
    my_sudoku, cols, blocks = s, [[s[j][i] for i in range(ROWS)] for j in range(ROWS)], convert_to_block(s)
    return has_no_dupes([*my_sudoku, *cols, *blocks])

def make_list(lst=None):
    return list([0 for i in range(ROWS)] for j in range(ROWS))


play_set = set(range(1, 10))    # {1, 2, 3, 4, 5, 6, 7, 8, ROWS}
all_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, ROWS}
slot_set = {0, 1, 2, 3, 4, 5, 6, 7, 8}
ref, REFP, lookups = generate_lookup()
table = {0: 'rows', 1: 'cols', 2: 'blocks'}
table2 = {'rows': 0, 'cols': 1, 'blocks': 2}
ref_xy = [[lookups[0][_idx * ROWS + _y] for _y in range(ROWS)] for _idx in range(ROWS*3)]
R, C, B, OT, OX, XY = 0, 1, 2, 3, 4, 5

# variables used in test functions below
board = [[1 + i + ROWS * j for i in range(ROWS)] for j in range(ROWS)]
board2 = make_list()

for i in range(ROWS):
    for j in range(ROWS):
        p, q = ref[(i, j)][table2['blocks']]  # change the original indices: i, j to blocks: p, q
        board2[p][q] = board[i][j]


def test_block_conversion():
    test_board = make_list()
    for i in range(ROWS):
        for j in range(ROWS):
            p, q = ref[(i, j)][table2['blocks']]  # change the original indices: i, j to blocks: p, q
            test_board[p][q] = board[i][j]
    assert convert_to_block(board) == test_board


def test_col_conversion():
    test_board = make_list()
    for i in range(ROWS):
        for j in range(ROWS):
            p, q = ref[(i, j)][table2['cols']]  # change the original indices: i, j to blocks: p, q
            test_board[p][q] = board[i][j]
    assert convert_rotate(board) == test_board


def test_col_lookups():
    row_list, col_list = [], []
    for i in range(ROWS):
        for j in range(ROWS):
            row_list.append((i, j))
            col_list.append((j, i))
    for col in col_list:
        assert col == ref[col][table2['rows']]
    for row in row_list:
        i, j = row
        assert (j, i) == ref[row][table2['cols']]


def test_block_lookups():
    row_list = []
    for i in range(ROWS):
        for j in range(ROWS):
            row_list.append((i, j))
    for row in row_list:
        i, j = row
        lookup_i, lookup_j = ref[row][table2['blocks']]
        assert board[lookup_i][lookup_j] == board2[i][j]

def test_generate_lookup():
    assert len(ref.keys()) == SIZE * 4
    my_list = list(product([i for i in range(ROWS)], repeat=2))
    for i in range(SIZE):
        assert len(set(my_list)) == SIZE
        assert 0 <= my_list[i][0] <= ROWS and 0 <= my_list[i][1] <= ROWS
        assert ref[my_list[i]]
        assert ref[("rows", *my_list[i])]
        assert ref[('cols', *my_list[i])]
        assert ref[('blocks', *my_list[i])]

        assert 0 <= ref[my_list[i]][0][0] <= ROWS and 0 <= ref[my_list[i]][0][1] <= ROWS
        assert 0 <= ref[my_list[i]][1][0] <= ROWS and 0 <= ref[my_list[i]][1][1] <= ROWS
        assert 0 <= ref[my_list[i]][2][0] <= ROWS and 0 <= ref[my_list[i]][2][1] <= ROWS

        assert 0 <= ref[('rows', *my_list[i])][0][0] <= ROWS and 0 <= ref[('rows', *my_list[i])][0][1] <= ROWS
        assert 0 <= ref[('rows', *my_list[i])][1][0] <= ROWS and 0 <= ref[('rows', *my_list[i])][1][1] <= ROWS
        assert 0 <= ref[('rows', *my_list[i])][2][0] <= ROWS and 0 <= ref[('rows', *my_list[i])][2][1] <= ROWS

        assert 0 <= ref[('cols', *my_list[i])][0][0] <= ROWS and 0 <= ref[('cols', *my_list[i])][0][1] <= ROWS
        assert 0 <= ref[('cols', *my_list[i])][1][0] <= ROWS and 0 <= ref[('cols', *my_list[i])][1][1] <= ROWS
        assert 0 <= ref[('cols', *my_list[i])][2][0] <= ROWS and 0 <= ref[('cols', *my_list[i])][2][1] <= ROWS

        assert 0 <= ref[('blocks', *my_list[i])][0][0] <= ROWS and 0 <= ref[('blocks', *my_list[i])][0][1] <= ROWS
        assert 0 <= ref[('blocks', *my_list[i])][1][0] <= ROWS and 0 <= ref[('blocks', *my_list[i])][1][1] <= ROWS
        assert 0 <= ref[('blocks', *my_list[i])][2][0] <= ROWS and 0 <= ref[('blocks', *my_list[i])][2][1] <= ROWS

        assert ref[my_list[i]][3][0][0] in ['rows', "cols", "blocks"] and ref[my_list[i]][3][1][0] in ['rows', "cols",
                                                                                                       "blocks"]
        assert 0 <= ref[my_list[i]][3][0][1] <= ROWS and 0 <= ref[my_list[i]][3][1][1] <= ROWS

        assert ref[('rows', *my_list[i])][3][0][0] in ['rows', "cols", "blocks"] and ref[('rows', *my_list[i])][3][1][
            0] in ['rows', "cols", "blocks"]
        assert 0 <= ref[('rows', *my_list[i])][3][0][1] <= ROWS and 0 <= ref[('rows', *my_list[i])][3][1][1] <= ROWS

        assert ref[('cols', *my_list[i])][3][0][0] in ['rows', "cols", "blocks"] and ref[('cols', *my_list[i])][3][1][
            0] in ['rows', "cols", "blocks"]
        assert 0 <= ref[('cols', *my_list[i])][3][0][1] <= ROWS and 0 <= ref[('cols', *my_list[i])][3][1][1] <= ROWS

        assert ref[('blocks', *my_list[i])][3][0][0] in ['rows', "cols", "blocks"] and \
               ref[('blocks', *my_list[i])][3][1][0] in ['rows', "cols", "blocks"]
        assert 0 <= ref[('blocks', *my_list[i])][3][0][1] <= ROWS and 0 <= ref[('blocks', *my_list[i])][3][1][1] <= ROWS


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
        y_list = choices(range(0,ROWS),k=randint(2,30))
        source, index = r1(), r2()
        ans = translate_list(source, index, y_list)
        assert len(y_list) == len(ans)
        for k in ans:
            assert 0 <= k[0] <= ROWS and 0 <= k[1] <= ROWS
            for i, y in enumerate(y_list):
                if source == "rows":
                    assert (index, y) == ans[i]
                if source == "cols":
                    assert (y, index) == ans[i]
                if source == "blocks":
                    assert board2[index][y] == board[ans[i][0]][ans[i][1]]
    assert True

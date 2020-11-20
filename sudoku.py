import inspect
import copy
from itertools import permutations, combinations
from sudoku_utils import *

class SudokuData:
    rows: list
    cols: list
    blocks: list
    ref: dict

    def __init__(self, slots: list=None):
        self.rows = list([0 for x in range(ROWS)] for y in range(ROWS))
        self.cols = list([0 for x in range(ROWS)] for y in range(ROWS))
        self.blocks = list([0 for x in range(ROWS)] for y in range(ROWS))
        if not slots:
            print("Empty SudokuData")
            return
        for x in range(ROWS):
            for y in range(ROWS):
                a, b = ref[(x, y)][2]
                self.rows[x][y] = self.cols[y][x] = self.blocks[a][b] = slots[x][y]
        self.ref = {"rows": self.rows, "cols": self.cols, "blocks": self.blocks}

    def add_entry(self, entry):
        z, x, y = entry
        if self.rows[x][y] and self.rows[x][y] != z:
            print("Error: Collision. Previously played entry found:", self.rows[x][y], "vs", z, "@", x, y)
            assert False
        elif self.rows[x][y] == z:
            print("Duplicate Entry @", x, y, z, "xyz", self.rows[x][y], "vs", z)
            return True
        bx, by = ref[(x, y)][2]
        if z in self.rows[x] or z in self.cols[y] or z in self.blocks[bx]:
            print("Error: Collision. With rows/cols/block mismatch")
            assert False
        self.cols[y][x] = self.rows[x][y] = z
        _p, _q = ref[(x, y)][2]
        self.blocks[_p][_q] = z
        return True

    def draw(self): print_sudoku(self.rows)

    def get_progress(self):
        return sum(x > 0 for index in range(ROWS) for x in self.rows[index])

    def get_plays(self, x, y):
        b_idx = ref[(x, y)][2][0]  # b_idx = block index
        return list(play_set - set(self.rows[x] + self.cols[y] + self.blocks[b_idx]))

    def open_slots(self):
        slots = []
        for _i in range(ROWS):
            for _j in range(ROWS):
                if self.rows[_i][_j] == 0:
                    slots.append((_i, _j))
        return slots

    def validate(self): return has_no_dupes([*self.rows, *self.cols, *self.blocks])

    def check_entry(self, z, x, y):
        block_index = ref[(x, y)][B][0]
        return self.rows[x][y] == 0 and 0 < z <= ROWS and \
               z not in self.rows[x] and z not in self.cols[y] and z not in self.blocks[block_index]

    def check_solved(self):
        r, c, b = self.rows, self.cols, self.blocks
        return all(play_set == set(r[_i]) == set(c[_i]) == set(b[_i]) for _i in range(ROWS))


# noinspection PyMethodMayBeStatic
class Group:
    source: str
    idx: int
    array: [list]

    count: int
    index: int
    slots: [int]
    xy: list
    mirrors: list
    m_groups: list
    m_slots: list
    reduced: list
    permutes: list
    g_slots: list
    g_plays: list

    def __init__(self, array: list, idx):
        self.array = array
        self.source = table[idx // ROWS]
        self.idx = idx
        self.slots = array[idx]
        self.xy, self.mirrors, self.m_groups = [], [], []
        self.g_slots, self.g_plays = [], []
        self.permutes = []
        for _k in range(ROWS):
            fields = REFP[(idx, _k)]
            self.xy += [fields[XY]]
            self.mirrors += [fields[OT]]
            self.m_groups += [fields[OX]]

    def group_info(self):
        print("Group: ", self.source, self.idx, "#", self.slots, "count/slots:", self.free_count(), self.free_slots())

    def info(self, iy):
        print("Info: (i.y.xy)", self.idx, iy, self.xy[iy], "mirrors, m-groups:", self.mirrors[iy], self.m_groups[iy])

    def combined(self, i):
        return self.slots + self.array[REFP[(self. idx, i)][OX][0]] + self.array[REFP[(self.idx, i)][OX][1]]

    def plays(self): return set(self.slots) - {0}

    def playable(self): return play_set - self.plays()

    def free_count(self): return len(self.playable())

    def free_slots(self): return [_iy for _iy in range(ROWS) if self.slots[_iy] == 0]

    def get_free_slot(self, num):
        if num >= self.free_count():
            return []
        return REFP[(self.idx,[_iy for _iy in range(ROWS) if self.slots[_iy] == 0][num])][XY]

    def validate(self):
        return len(self.plays() | self.playable()) == 9 and \
               ROWS - len(self.plays()) == self.free_count() and self.array[self.index] == self.slots

    def translate(self, ix, z: int=None):
        return REFP[(self.idx, ix)][0] if not z else z + REFP[(self.idx, ix)][0]

    def gen_plays(self):   # returns the lowest number of variations of play (set)
        slots = self.g_slots = self.free_slots()
        if not self.g_slots:
            self.g_plays = []
            return []
        plays = self.g_plays = [list(play_set - set(self.combined(_k))) for _k in self.g_slots]
        self.reduced = sorted(((plays[_i], REFP[(self.idx,_s)][XY]) for _i, _s in enumerate(slots)), key=lambda _p: len(_p[0]))
        return len(self.reduced[0][0])

    # currently  unused
    def slots4play(self, _play):
        if _play not in self.playable():
            return []
        self.gen_plays()
        return [REFP[(self.idx, *self.g_slots[_i])][XY] for _i in range(self.free_count()) if _play in self.g_plays[_i]]


    def find_freq(self):
        self.gen_plays()
        pot = [play for g in self.g_plays for play in g]
        plays = list(self.playable())
        frequencies, tb = [], [0]*len(plays)
        for _p in pot:
            tb[plays.index(_p)] += 1
        return sorted(zip(plays, tb))

    def find_permute(self):
        free_slots = self.free_slots()
        free_plays = self.playable()
        count = len(free_plays)

        play_pack = []
        for _k in free_slots:
            free_play = play_set - set(self.combined(_k))
            play_pack.append(free_play)

        self.permutes = []
        permute_ok = []
        for permuted in permutations(free_plays, count):
            walk = 0
            while walk < count:
                if not permuted[walk] in play_pack[walk]:
                    break
                walk += 1
            if walk == count:
                permute_ok.append(permuted)
        self.permutes = [[(_play, *REFP[(self.idx, free_slots[_i])][XY]) for _i, _play in enumerate(p_set)] for p_set in permute_ok]

        if len(self.permutes) == 1:
            return True
        return False

    def find_partial(self, reload: bool = True):
        if reload and self.find_permute():
            return self.permutes[0]
        elif len(self.permutes) < self.free_count():
            return []

        count = self.free_count()
        matched, matched_pack = [], []
        if len(self.permutes) < count:
            for _i in reversed(range(count)):
                if all(self.permutes[0][_i][0] == perm[_i][0] for perm in self.permutes[1:]):
                    matched.append(_i)
                    for repair in self.permutes:
                        repair.pop(_i)
            if matched:
                return [(self.permutes[0][_i][0], *self.permutes[0][_i][1:]) for _i in matched]
        return []

class Trial(SudokuData):
    original: SudokuData
    current: SudokuData
    array: list
    groups: [Group]
    play_records: list
    log: list
    rows: list
    cols: list
    blocks: list
    permuted: list

    # noinspection PyMissingConstructor
    def __init__(self, game: SudokuData):
        self.stop = ROWS*3
        self.original = copy.deepcopy(game)
        self.current = copy.deepcopy(game)
        self.rows, self.cols, self.blocks = self.current.rows, self.current.cols, self.current.blocks
        self.ref = self.current.ref
        self.array = [*self.rows, *self.cols, *self.blocks]
        self.groups = [Group(self.array, _i) for _i in range(ROWS * 3)]

        self.log = []
        self.play_records = []
        self.queue = []
        self.trials = []

    def logging(self, log_msg: str= ""):
        self.log.append(log_msg+" - " + str(inspect.stack()[1][3]))

    def add_entry(self, entry):
        self.current.add_entry(entry)
        self.play_records.append((entry, inspect.stack()[1][3]))

    def check_trial(self):
        if not self.original.validate():
            print("\nERROR: Original Sudoku is invalid.\n")
            return False
        if not self.validate():
            print("\nERROR: Trial Sudoku is invalid.\n")
            return False
        org, test = self.original.rows, self.rows
        if not all(org[x][y] == 0 or org[x][y] == test[x][y] for x in range(ROWS) for y in range(ROWS)):
            print("\nERROR: Invalid Sudoku: Violating original start.\n")
            return False
        return True

    def auto_solve(self, trial=0):
        channel_check = "update"
        self.logging("trial auto_solve begins")
        if self.check_solved():
            print("* ** Solved! ** *")
            self.draw()
            return SIZE
        last_status, loop_counter = 0, 0
        while last_status != self.get_progress():
            last_status = self.get_progress()
            self.try_sets()
            if self.get_progress() == SIZE:
                print("* ** Solved! ** *")
                self.draw()
                assert self.check_solved() and self.check_trial()
                print("Final check:", self.check_trial())
                return SIZE
            elif last_status == self.get_progress():
                print("try permutes")
                self.try_permutes()
                self.try_frequencies()
            else:
                print("Current Progress:", self.get_progress(), self.get_progress() * 1000 // SIZE / 10, "%")
                self.draw()
                self.logging("Main Loop Completed #"+str(loop_counter))
            loop_counter += 1
            # Main loop ends
        print()
        print("Quitting trial auto_solve: final check:", self.check_trial())
        if not self.check_solved():
            printList(self.log,"Log: Functions used:")
            printList(self.play_records,"Log: Play Records:")
            print("Final Progress:", self.get_progress(), self.get_progress() * 1000 // SIZE / 10, "%")

        return self.get_progress()

    def try_frequencies(self):
        self.logging()
        frequencies = [(fr[0], ROWS - fr[1]) for fr in freq(self.rows) if fr[0] != 0 and fr[1] != 0]
        frq = copy.deepcopy(frequencies)
        while frq:
            if frq[0][1] == 0:
                frq.pop(0)
                continue
            elif frq[0][1] < 0:
                self.logging("Error. Play: "+str(frq[0][0])+" exceed allowed count 9: "+str(frq[0][1]))
                print("Error. Play: "+str(frq[0][0])+" exceed allowed count 9: "+str(frq[0][1]))
                assert False
            filtered = list(filter(lambda _k: self.check_entry(frq[0][0], *_k), self.open_slots()))
            if not filtered:
                if frq[0][1] == 0:
                    printList(self.log[-5:])
                    printList(self.play_records[-5:])
            if filtered and len(filtered) == frq[0][1] and len(filtered) < 5:
                print_sudoku(self.rows)
                while filtered:
                    self.add_entry((frq[0][0], *filtered[0]))
                    filtered.pop(0)
                print_sudoku(self.rows)
            elif filtered:
                # print(f"Warning: Play:{frq[0][0]} @freq:{frq[0][1]}x with {len(filtered)} slots.")
                if len(filtered) < frq[0][1]:
                    print("Less slots than frequency. Possible Invalid Sudoku.")
            frq.pop(0)
        self.logging()
        return self.get_progress()

    def try_permutes(self):

        groups = sorted(self.groups, key=lambda grp: grp.free_count())
        groups = [grp for grp in groups if 0 != grp.free_count() != 2]

        total = len(groups)
        factor = total//20 + total//14
        p_groups = groups[: total // (1 + factor)]
        walk = 0
        self.logging()
        while True:
            while walk < len(p_groups):
                free_count = p_groups[walk].free_count()
                previous = len(p_groups)
                if free_count == 0:
                    p_groups.pop(walk)
                elif free_count == 1:
                    (play,), (x, y) = p_groups[walk].playable(), p_groups[walk].get_free_slot(0)
                    self.add_entry((play, x, y))
                    p_groups.pop(walk)
                elif free_count == 2:
                    while p_groups[walk].gen_plays() == 1:
                        [(play,), (x, y)] = p_groups[walk].reduced[0]
                        self.add_entry((play, x, y))
                    if p_groups[walk].free_count() == 2:
                        p_groups.pop(walk)
                else:
                    if p_groups[walk].find_permute():
                        for entry in p_groups[walk].permutes[0]:
                            self.add_entry(entry)
                        p_groups.pop(walk)
                    else:
                        entries = p_groups[walk].find_partial()
                        if entries:
                            for entry in p_groups[walk].permutes[0]:
                                self.add_entry(entry)
                            break
                if previous != len(p_groups):
                    break
                walk += 1
            if walk >= len(p_groups) and factor:
                half = total // (1 + factor)
                p_groups = groups[half:]
                factor = 0
                walk = 0
            elif walk >= len(p_groups):
                break
        # print("leaving try_permute")
        self.logging()
        return self.get_progress()

    def try_sets(self):
        self.logging("enter")
        while True:
            idx, pairs = 0, []
            while idx < len(self.groups):
                grp = self.groups[idx]
                if grp.free_count() == 0 and idx in pairs:
                    idx += 1
                    continue
                while grp.gen_plays() == 1:
                    [(play,), (x, y)] = grp.reduced[0]
                    self.add_entry((play, x, y))
                if grp.free_count() == 2:
                    pairs.append(idx)
                    idx += 1
                    if grp.g_plays[0] != grp.g_plays[1]:
                        assert False
                    continue
                playable = remove_dupes(list(map(set, grp.g_plays)))
                while playable:
                    play = playable[0][0]
                    x, y = grp.translate(grp.g_slots[playable[0][1]])
                    self.add_entry((play, x, y))
                    playable.pop(0)
                idx += 1
            if idx >= len(self.groups):
                break
        self.logging("leaving")

class Game(SudokuData):
    sudoku: SudokuData
    trials: [Trial]
    log: list

    # noinspection PyMissingConstructor
    def __init__(self, file_name: str =None):
        if not file_name:
            _slots = self.load_file("games\\s01a.txt")
            self.sudoku = SudokuData(_slots)
        else:
            _slots = self.load_file(file_name)
            self.sudoku = SudokuData(_slots)
        self.rows, self.cols, self.blocks = self.sudoku.rows, self.sudoku.cols, self.sudoku.blocks
        self.ref = self.sudoku.ref
        self.trials = []

    def add_entry(self, entry):
        self.sudoku.add_entry(entry)

    @staticmethod
    def load_file(file_name):
        list_of_lists = []
        print(file_name)
        with open(file_name) as f:
            for _i, line in enumerate(f):
                if _i == ROWS:
                    break
                line = line[:-2]
                inner_list = [int(elt.strip()) for elt in line.split(' ')]
                # in alternative, if you need to use the file content as numbers
                # inner_list = [int(elt.strip()) for elt in line.split(',')]
                list_of_lists.append(inner_list)

        return list_of_lists[0:ROWS]

    def print(self):
        print_sudoku(self.sudoku.rows)

    def add_trial(self, slots=None, starter_list=None):
        # Can base on the original(s=None) or orig-variant or entirely-new
        if not slots:
            self.trials.append(Trial(self.sudoku))
            return
        self.trials.append(Trial(slots))

    def auto_solve(self, game_id=0):
        self.add_trial()
        print("game autosolve")
        return self.trials[game_id].auto_solve()


def remove_dupes(list_sets):
    l_set = copy.deepcopy(list_sets)
    count = len(list_sets)
    frequencies = freq(l_set, reverse=False)
    indices = list(range(count))
    playable = []
    exclude = []

    while frequencies:
        play, times = frequencies[0]
        if times == 1:
            p_index = 0
            for _j in range(len(l_set)):
                if play in l_set[_j]:
                    indices.remove(_j)
                    p_index = _j
                    break
            playable.append((frequencies[0][0], p_index))
            l_set = list(map(lambda set_plays: set_plays - {play}, l_set))
            # print("FOUND, playable in freq:", play)
        else:
            break
        frequencies.pop(0)

    while any(len(l_set[i]) == 1 for _i in indices if _i not in indices):
        for _i in indices:
            if len(l_set[_i]) == 1:
                (_play,) = l_set[_i]
                playable.append((_play, _i))
                l_set = list(map(lambda set_plays: set_plays - {_play}, l_set))
            if not l_set[_i] and _i in indices:
                indices.remove(_i)

    combined = [[] for _ in range(count)]
    if len(indices) > 2:
        for _i in indices:
            i_pool = list(set(indices) - {_i})
            combined[_i] = [_c for _j in range(1, len(indices)) for _c in combinations(i_pool, r=_j)]
        for _i in indices:
            for combo_i in combined[_i]:
                combo_set = set().union(*(l_set[each_i] for each_i in list(combo_i)))
                if not combo_set:
                    continue
                if len(combo_set) == len(combo_i):
                    exclude += list(combo_i)
                    xor_set = l_set[_i] - combo_set
                    if len(xor_set) == 1:
                        (_play,) = xor_set
                        playable.append((_play, _i))
    if len(playable) < 2:
        return playable
    playable = list(set(playable))
    if any(playable[0][0] == _play[0] for _play in playable[1:]):
        print("Error: remove_dupes. Duplicates. Same play at different slots. Each play should be unique.")
    return playable

def load_file(file_string):
    list_of_lists = []
    with open(file_string) as f:
        for line in f:
            #
            # inner_list = [int(elt.strip()) for elt in line.split(' ')]
            list_of_lists.append(line[:-1])
    return list_of_lists


wins, stalled, starting, games_progresses, game_files = 0, 0, [], [], []
file_list = load_file("game_list.txt")

for k in range(0, 17):
    print()
    my_game = Game(file_list[k])
    start_stat = my_game.get_progress()
    reply = my_game.auto_solve()
    if reply == SIZE:
        wins += 1
    else:
        print("game replied:", reply)
        stalled += 1
        starting.append(start_stat)
        games_progresses.append(reply)
        game_files.append(file_list[k])
    print("Game Solved:", wins, "against stalled:", stalled)
results = [f"Played: {g - starting[i]} - {(SIZE - g)*10000//81/100}% .." for i, g in enumerate(games_progresses)]

printList(results,"Auto-solve Stalled Games Report")


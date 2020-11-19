# Sudoku Game - Sudoku Checker
# 1. Sudoku game is a class that takes an initial array with existing elements that form initial puzzle.
#        It consists of 2-D array [0:9][0:9] initialized to 0 at empty spots.
# 2. It has a randomly created sudoku maker which initialize the sudoku with randomly initialized elements
# 3. It has a checker to check whether a sudoku with added input is valid
import copy
# from enum import Enum
from dataclasses import dataclass, field
from itertools import chain, permutations, combinations, takewhile
from operator import itemgetter
from random import shuffle
from typing import List, Type, Callable

def generate_lookup():
    keys, values = [], [[] for i in range(4)]
    for i in range(9):
        for j in range(9):
            keys.append((i, j))    # first 81 keys
            values[0].append((i, j))  # first 81 rows
            values[1].append((j, i))  # first 81 cols
    for i in range(9):
        for j in range(9):    # 2*81 key: row
            keys.append(('rows', i, j))
            values[0].append((i, j)) # 160
            values[1].append((j, i)) # 160
    for i in range(9):        # 3*81 key: col
        for j in range(9):
            keys.append(('cols', j, i))
            values[0].append((i, j))   # 270
            values[1].append((j, i))   # 270
    for i in range(81):    # block first 81
        x, y = keys[i]
        p = (x // 3) * 3 + (y // 3)          # block-number
        q = (x % 3) * 3 + (y % 3)          # block-index
        values[2].append((p, q))   # block 81
    for i in range(81):
        values[2].append(values[2][i])   # block 80,160
    for i in range(81):
        x, y = values[2][i]
        keys.append(('blocks', x, y))
        values[0].append(values[0][i])
        values[1].append(values[1][i])
    for i in range(81):
        values[2].append(values[2][i])   # block 160, 270 (col)  # same lookup tables 0-81  since col 0-81 is block 0-81
    for i in range(81):
        values[2].append(values[2][i])    # block 270, 320 (block)
    pair = "cols", "blocks"
    for k in range(81 * 4):
        if (2*81) <= k < (3*81):
            pair = "rows", "blocks"
        elif (3*81) <= k < (4*81):
            pair = "rows", "cols"
        look = {'cols': 1, 'rows': 0, 'blocks': 2}
        values[3].append(((pair[0], values[look[pair[0]]][k][0]), (pair[1], values[look[pair[1]]][k][0])))
    lookup = {}
    test = list(zip(values[0], values[1], values[2], values[3]))
    for k in range(81 * 4):
        lookup[keys[k]] = test[k]
    return lookup

def test_args(func):
    def _wrapper(*args,**kwargs):
        print("Test func arguments: ", len(args), "args", len(kwargs), "kwargs")
        print("Args:", *args, **kwargs)
        return func(*args,**kwargs)
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

def test_collision_entries(slots):
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
    if source != 'blocks' and to != 'blocks':
        i, j = y, x
    elif source == "rows" or to == "rows":
        i, j = ((x // 3) * 3 + y // 3, (y % 3) + (x % 3 * 3))
    else:
        j, i = ((x // 3) * 3 + y // 3, (y % 3) + (x % 3 * 3))
    return i, j

def translate_list(source, x, y, to="rows"):
    if source == to:
        return [(x, j) for j in y]
    if source in ("rows", "cols") and to in ("rows", "cols"):
        return [(j, x) for j in y]
    temp_list = y
    if source == "rows" or to == "rows":
        return [((x // 3) * 3 + y // 3, (y % 3) + (x % 3 * 3)) for y in temp_list]
    return [((y % 3) + (x % 3 * 3), (x // 3) * 3 + y // 3) for y in temp_list]


def converter(func):
    def _wrap(*args, **kwargs) -> list:
        idx = func(*args, **kwargs)
        converted = [[0 for i in range (9)] for j in range(9)]
        for i in range(9):
            for j in range(9):
                p, q = ref[(i, j)][idx]
                converted[p][q] = args[0][i][j]
        return converted
    return _wrap

@converter
def convert_to_block(slots=None): return 2

@converter
def convert_to_original(slots=None): return 0

@converter
def convert_rotate(slots=None): return 1


def has_no_dupes(slots=None):
    for new_list in slots:
        count = sum(x > 0 for x in new_list)
        set_l = set(new_list)
        set_l.discard(0)
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


play_set = {1, 2, 3, 4, 5, 6, 7, 8, 9}
all_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
slot_set = {0, 1, 2, 3, 4, 5, 6, 7, 8}
ref = generate_lookup()


@dataclass
class Chan:
    source: str = "rows"
    index: int = 0
    slots: list = field(default_factory=list)

@dataclass
class Channel(Chan):
    id: int = 99        # unspecified
    count: int = 0
    filled: list = field(default_factory=list)
    bids: list = field(default_factory=list)
    queue: list = field(default_factory=list)
# Generate min-effort fill-list(s)** (max-occupancy)
# Need Restructuring:  count, source, index, slots, bids(later) + queue? low-occupancy (more-slot/branching)

class SudokuData:
    rows: list
    cols: list
    blocks: list
    ref: dict

    def __init__(self, slots: list=None):
        self.cols = list([0 for i in range(9)] for j in range(9))
        self.blocks = list([0 for i in range(9)] for j in range(9))
        if not slots:
            print("Empty SudokuData")
            self.rows = list([0 for i in range(9)] for j in range(9))
            return
        self.rows = copy.deepcopy(slots)
        for i in range(9):
            for j in range(9):
                self.blocks[(i//3) * 3 + j//3][(j % 3) + (i % 3 * 3)] = self.cols[j][i] = self.rows[i][j]
        self.ref = {"rows": self.rows, "cols": self.cols, "blocks": self.blocks}


class Sudoku(SudokuData):
    saved: list

    def __init__(self, slots: List=None, sudoku=None):
        super().__init__(slots)
        print("Sudoku created")
        self.saved = make_list()

    def update(self, slots):
        if not check_sudoku(slots):
            print("Error creating Sudoku. Wrong Input.")
        for i in range(9):
            for j in range(9):
                self.blocks[(i // 3) * 3 + j // 3][(j % 3) + (i % 3 * 3)] = self.cols[j][i] = self.rows[i][j] = slots[i][j]

    def add_entry(self, x, y, z, trial=0):
        if self.check_entry(x, y, z):
            self.rows[(x // 3) * 3 + y // 3][(y % 3) + (x % 3 * 3)] = self.cols[y][x] = self.saved[x][y] = z
            return True
        return False

    def check(self):   # check entire sudoku
        return has_no_dupes(slots=[*self.rows, *self.cols, *self.blocks])

    def check_entry(self, x, y, z):
        def block_index(k): return (k[0] // 3) * 3 + k[1] // 3
        if self.rows[x][y] != 0 or not 0 < z <= 9 or \
                z in self.rows[x] or z in self.cols[y] or z in self.blocks[block_index((x, y))]:  # Redundant check
            return False
        return True

    def check_entries(self, entries: List):
        return all(self.check_entry(x, y, z) for (x, y, z) in entries)

    def print(self, slots=None):
        if slots:
            print_sudoku(slots)
            return
        print_sudoku(self.rows)

class Game:
    def __init__(self, slots: List=None):
        self.starter = Sudoku(slots=slots)
        self.games = []   # variants built-upon the starter
        self.games.append(Trial(slots=slots))  # original variant   gives self.starter (Sudoku) to

    def add_trial(self, slots=None, starter_list=None):
        # Can base on the original(s=None) or orig-variant or entirely-new
        if not slots:
            self.games.append(Trial(slots=self.starter.rows, tryout_list=starter_list))
            return
        self.games.append(Trial(slots=slots, tryout_list=starter_list))

    def auto_solve(self, game_id=0):
        return self.games[game_id].auto_solve()

    @staticmethod
    def load_file(file_str):
        list_of_lists = []
        print(file_str)
        with open(file_str) as f:
            for i, line in enumerate(f):
                if i == 9:
                    break
                line = line[:-2]
                inner_list = [int(elt.strip()) for elt in line.split(' ')]
                # in alternative, if you need to use the file content as numbers
                # inner_list = [int(elt.strip()) for elt in line.split(',')]
                list_of_lists.append(inner_list)

        return list_of_lists[0:9]

    def load(self, file_str):
        list_of_lists = Game.load_file(file_str)
        self.starter.update(s=list_of_lists)
        return list_of_lists

    def get_sudoku(self):
        return self.games[-1]


class Trial:
    def __init__(self, slots, tryout_list=None):
        self.checker = {'rows': self.row_check, 'cols': self.col_check, 'blocks': self.block_check}
        self.work_path = []
        self.seed = []
        self.queue = []
        self.trials = []
        self.original = SudokuData(slots)
        self.trials.append(SudokuData(slots))
        if tryout_list:
            self.seed.append(tryout_list)

    def add_trial(self):
        print("Added new trial")
        self.trials.append(SudokuData(self.rows))
        return len(self.trials) - 1

    def check_trial_sudoku(self, trial: int=0):
        board = self.trials[trial]
        if not self.check_original(trial):
            return False
        if not check_sudoku(board.rows):
            print("\nERROR: Invalid Sudoku: Trial No.", trial, "\nInvalid entries on board.\n")
            return False
        if not check_sudoku(convert_to_original(board.blocks)) or not check_sudoku(convert_rotate(board.cols)):
            print("\nERROR: Invalid Sudoku: Trial No.", trial, "\nBad variants. (Rows/Cols)\n")
            return False
        return True

    def check_original(self, trial: int = 0):
        if not check_sudoku(self.trials[trial].rows):
            print("\nERROR: Invalid Sudoku: Bad entries on board.\n")
            return False
        org, test = self.original.rows, self.trials[trial].rows
        if not all(org[i][j] == 0 or org[i][j] == test[i][j] for i in range(9) for j in range(9)):
            print("\nERROR: Invalid Sudoku: Violating original start.\n")
            return False
        return True

    def check_final(self, trial: int = 0):
        if self.check_trial_sudoku(trial):
            s = self.trials[trial]
            if all(45 == sum(s.rows[i]) == sum(s.cols[i]) == sum(s.blocks[i]) for i in range(9)):
                print("Sudoku Final Check: Passed! Game is solved!")
                return True
        return False

    def update_processing(self, slots, plays, trial: int=0):
        slot_pairs = translate_list(*slots)
        return all(self.update_entry((x, y, plays[i]), trial) for i, (x, y) in enumerate(slot_pairs))

    def update_process_entry(self, slot, play, trial: int=0):
        slot = slot if type(slot[2]) is int else (slot[0], slot[1], slot[2][0])
        x, y = translate(*slot)
        return self.update_entry((x, y, play),trial)

    def update_entries(self, entries, trial: int=0):
        return all(self.update_entry(entry, trial) for entry in entries)

    def update_entry(self,entry, trial: int=0, check=True):
        x, y, z = entry
        board = self.trials[trial]
        if board.rows[x][y] and check and board.rows[x][y] != z:
            print("Error: Previously filled play: ", (x, y, board.rows[x][y]), "vs", entry)
            return False
        # print("Debug: Adding entry:", entry)
        board.cols[y][x] = board.rows[x][y] = z
        board.blocks[(x // 3) * 3 + y // 3][(y % 3) + (x % 3 * 3)] = z
        return True

    def auto_solve(self, trial=0):
        if not self.trials:
            self.add_trial()
        count = 1
        channel_check = True
        test = self.trials[trial]
        if not has_no_dupes(slots=[*test.rows, *test.cols, *test.blocks]):
            print()
            print("ERROR START")
            print()
        while channel_check != False and channel_check != "solved":
            if self.check_trial_sudoku(trial):
                rate = check_progress(self.trials[trial].rows)
                # print("\nProgress! Update #", count, "# Block filled =", rate, (rate/81)*10000//100,"%")
                print_sudoku(self.trials[trial].rows)
            else:
                print_sudoku(self.trials[trial].rows)
                print("Failed Sudoku check\n")
            channel_check = self.try_channels(trial)
            count += 1
        if channel_check == "solved":
            print("Sudoku is solved.\n")
            return True
        return False

    def try_channels(self, trial=0):
        # Approach #1 - explore trials for each block (row/col)
        channels = self.generate_channels()
        idx = 0
        while idx < 27 and len(channels[idx][2]) == 0:  # discard filled slots
            idx += 1
        filled = idx
        if idx >= 27:
            print("Sudoku solved????")
            return "solved"

        # Fill single slots
        while idx < 27 and len(channels[idx][2]) == 1:
            play = 45 - sum(self.trials[trial].ref[channels[idx][0]][channels[idx][1]])
            if play == 0:
                print("The block/row/col is full", channels[idx], translate(channels[idx][0], channels[idx][1], channels[idx][2][0]))
                idx += 1
                continue
            # print("play this: ", channels[idx], play)
            if not self.update_process_entry(channels[idx], play, trial):
                print("Check additions :", channels[idx], play)
                return False
            idx += 1
        if idx - filled > 0:
            return True
        elif idx >= 27:
            print("Sudoku is solved")
            return "solved"

        # Test lowest One-Third or One-Half : slice by len//2  (for >17) or len //3 (for > 21)  # Can Tune further
        valid_channels = len(channels) - idx
        divisor = 1 + valid_channels//24 + valid_channels//19
        if divisor > 1:
            stop = (valid_channels // divisor)+idx
        else:
            stop = len(channels)
        print("stop/queue @ channel no.", idx, stop)
        shuffled_channels = channels[idx:stop]
        self.queue = channels[stop:]    # save the rest in queue
        shuffle(shuffled_channels)
        # printList(shuffled_channels,"Sliced shuffled chans")

        # generate tryouts --- (set) of plays (fill entire block/row/col)
        tryouts = []
        for ch in shuffled_channels:
            tryouts.append(play_set-set(self.trials[trial].ref[ch[0]][ch[1]]))
        print("No of tryouts",len(tryouts))
        best_tryouts = []
        no_solutions = []
        for i,ch in enumerate(shuffled_channels):
            # one block at a time:
            variants = permutations(tryouts[i],len(tryouts[i]))
            good_variants = list(variant for variant in variants if self.check_plays(ch[0],ch[1],ch[2],variant))
            # A new approach: can only check one slot at a time
            playable_flag_flag = True
            for slot in ch[2]:
                # lookup, get the row and col indices (to detect collision) ensure no detection
                keys= ['rows', 'cols', 'blocks'] ; keys.remove(ch[0])
                r, c = self.trials[trial].ref[keys[0]], self.trials[trial].ref[keys[1]]
                ri, ci = ref[(ch[0], ch[1], slot)][3][0][1], ref[(ch[0], ch[1], slot)][3][1][1]
                playable = play_set - set(r[ri]) - set(c[ci])
                if playable.intersection(set(tryouts[i])):
                    pass   # print("OK", i)
                else:
                    playable_flag_flag = True
                    break
            else:
                playable_flag_flag = True

            if len(good_variants):
                best_tryouts.append((ch,good_variants))
            else:
                print("\nNo viable solution for this:", i, shuffled_channels[i])
                no_solutions.append(i)
        best_tryouts = sorted(best_tryouts, key=lambda candidate: (len(candidate[1]), len(candidate[0][2])))
        # printList(best_tryouts, "Best Tryouts")

        proven_trial = 0
        while proven_trial < len(best_tryouts) and len(best_tryouts[proven_trial][1]) == 1:   # Jackpot
            proven_trial += 1
        if proven_trial > 1:
            print("Possible Problem: more than 1 solution.", proven_trial)  # Possible way to check solution is not possible
            # printList(best_tryouts[:proven_trial], "Best Tryouts. Multiple single set fills.")
            slots, plays = [], []
            for i in range(proven_trial):
                plays.append([(*k, best_tryouts[i][1][0][j]) for j, k in enumerate(translate_list(*best_tryouts[i][0]))])
            entire_pot = test_collision_entries(chain.from_iterable(plays))
            if not entire_pot:
                print("Collision error: possible invalid sudoku")
                return False
            return self.update_entries(entire_pot, trial)
        elif proven_trial == 1:
            self.update_processing(best_tryouts[0][0], (best_tryouts[0][1][0]))
            return "update"
        print("Branching encountered")

        # Find common element among each trial and update trial
        entries = []     # Find overlaps = the slot must fill
        for tryout in best_tryouts:
            indices = len(tryout[1][0])
            groups = len(tryout[1])
            play_test = tryout[1]
            if indices == 2:
                continue
            index_tally = []
            for i in range(indices):
                for k in range(1, groups):
                    if not play_test[0][i] == play_test[k][i]:
                        break
                else:
                    print("All matching index:",i, "k-check",k)
                    index_tally.append(i)
            if len(index_tally)>0:
                print("Tally", index_tally, "groups, indices", groups, indices)
            for j in index_tally:
                entries.append((*translate(*tryout[0][:2], tryout[0][2][j]), tryout[1][0][j]) )
        if entries:   # Check the common overlaps <= we can fill these
            # printList(entries+best_tryouts, "Common overlaps cross-matched. We can fill these. (Progress!)")
            entries = test_collision_entries(entries)
            if not entries:
                print("Collision error: possible invalid sudoku")
                return False
            print("About to update")
            self.update_entries(entries, trial)
            print("updated")
            return "update"
        print("No Common elements. True branching starts here.")
        print("Make sure everything works. Is the Sudoku Valid?", "yes" if self.check_trial_sudoku(trial) else "no")
        printList(best_tryouts, "best-tryouts left?")
        print("No of tryouts",len(best_tryouts))

        return False

    def generate_channels(self, trial=0):    # which data-format to use???
        # Now that code is more organized, I am still trying to figure out which data format to use
        # Generate min-effort fill-list(s)** (max-occupancy)
        # Need Restructuring:  count, source, index, slots, bids(later) + queue? low-occupancy (more-slot/branching)
        counter = lambda slots: sum(x > 0 for x in slots)
        tries = self.trials[trial]
        all_ch = list(chain.from_iterable((("rows", i, [j for j in range(9) if not tries.rows[i][j]]),
                                           ("cols", i, [j for j in range(9) if not tries.cols[i][j]]),
                                           ("blocks", i, [j for j in range(9) if not tries.blocks[i][j]]))
                                          for i in range(9)))    # all_ch:  empty-slots
        channels = sorted(all_ch, key=lambda ch: len(ch[2]))
        return channels

    def filter_collisions(self, prospects):
        # OK approach. Probably not that great. Very messy now. If needed, we can clean up and get it to work.
        # Generate collision-free fewest-branch candidate-lists per fill-list* (each)
        pro_list = []
        for (source, index) in prospects:
            #  highest filled row/col/block is being tried; find which spots are empty;
            prospect = play_set - set(self.ref[source][index])
            y_spots = list(i for (i, k) in enumerate(self.ref[source][index]) if k == 0)
            # We have row/col/block(x) index(y)(empty-spots) candidates(Ref{1..9})
            bids = permutations(prospect)  # possible bids (permuted to try)
            print("##", index, source)
            passed = list(filter(lambda bid: self.box_checks(source, index, y_spots, bid), bids))
            if len(passed) > 0:
                pro_list.append((len(passed), source, index, y_spots, passed))
        min2, _, _, _, _ = min(pro_list, key=itemgetter(0))

        # each list is a block/row/col with possible set of fillers(s); min2==1 best possible (seed)
        pro_list2 = list(filter(lambda p2: p2[0] == min2, pro_list))

    def check_plays(self, source, index, y_list, bids, trial=0):
        if not len(y_list)== len(bids):
            print("\n\n#  ERROR - check_plays  # ")
        else:  # Code if we want to take in more plays than slots
            for i in range(len(y_list)):
                verify1, verify2 = True, True
                a, b = ref[(source, index, y_list[i])][3]
                p, q = ref[(source, index, y_list[i])][0]
                r = ref[(p, q)][2][0]
                verify1 = bids[i] in self.trials[trial].ref[a[0]][a[1]] or bids[i] in self.trials[trial].ref[b[0]][b[1]]
                verify2 = bids[i] in self.trials[trial].rows[p] or bids[i] in self.trials[trial].cols[q]
                verify2 &=  bids[i] in self.trials[trial].ref["blocks"][r]
                if self.trials[trial].rows[p][q] == bids[i]:
                    print(" error - check_play: filled")
                    return False
                if verify1 != verify2:
                    print("Old test failed. r/c. T/F, s,i,y - x,y :", verify1, source, index, y_list[i], p, q)
                if not verify1:
                    return verify1  # For now, we'll stick with the old ways
            return True

    def box_check(self, source, index, y, bid, trial=0):
        func = {'rows': self.row_check, 'cols': self.col_check, 'blocks': self.block_check}
        return func[source](index, y, bid, trial)

    def box_checks(self, source, index, y_list, bids, trial=0):
        func = {'rows': self.row_check, 'cols': self.col_check, 'blocks': self.block_check}
        if not len(y_list)== len(bids):
            print("\n\n#$  ERROR   ")
            print(source,index,y_list, bids)
            print_sudoku(self.trials[trial].rows)
            printList(self.trials[trial].blocks)
            print(len(y_list), len(bids) )
        return all(func[source](index, y_list[k], bids[k], trial) for k in range(len(bids)))

    def col_check(self, index, y, bid, trial=0):  # check rows = col.index=j y=x
        test, row, blk = True, self.trials[trial].rows, self.trials[trial].blocks
        k1, k2 = ref[("rows", index, y)][3]
        k1, k2 = k1[1], k2[1]
        if bid in self.trials[trial].rows[y] or bid in self.trials[trial].blocks[(y // 3) * 3 + index // 3]:
            test = False
        new_test = bid in row[k1] or bid in blk[k2]
        if test != new_test:
            pass
        return not new_test

    def row_check(self, index, y, bid, trial=0):  # check rows = col.index=j y=x
        test, col, blk = True, self.trials[trial].cols, self.trials[trial].blocks
        k1, k2 = ref[("rows", index, y)][3]
        k1, k2 = k1[1], k2[1]
        if bid in self.trials[trial].cols[y] or bid in self.trials[trial].blocks[(index // 3) * 3 + y // 3]:
            test = False
        new_test = bid in col[k1] or bid in blk[k2]
        if test != new_test:
            pass
        return not new_test

    def block_check(self, index, y, bid, trial=0):  # check rows = col.index=j y=x
        test, row, col = True, self.trials[trial].rows, self.trials[trial].cols
        k1, k2 = ref[("blocks", index, y)][3]
        k1, k2 = k1[1], k2[1]
        if bid in self.trials[trial].cols[(y // 3) * 3 + index // 3] \
                or bid in self.trials[trial].rows[(index // 3) * 3 + y // 3]:
            test = False
        new_test = bid in row[k1] or bid in col[k2]
        if test != new_test:
            pass
        return not new_test

def load_file(file_str):
    list_of_lists = []
    print(file_str)
    with open(file_str) as f:
        for line in f:
            #
            # inner_list = [int(elt.strip()) for elt in line.split(' ')]
            list_of_lists.append(line[:-1])
        else:
            list_of_lists.pop()
    return list_of_lists

# def main():
#     test = SudokuData()
#     print(dir(test),test)
# if __name__=="__main__":
#     main()
file = "sudoku\\s01b.txt"
file_list = "sudoku\\filelist"
loaded = load_file(file_list)
look = generate_lookup()
# printList([(k,look[k] )for k in look.keys() if k[0]=='cols'])
# k = ('cols',3,7)
# print(k,look[k])

wins, stalled = 0, 0
for k in range(0, 7):
    file_str = Game.load_file("sudoku\\"+loaded[k])
    my_game = Game(file_str)
    reply = my_game.auto_solve()
    if reply is True:
        wins += 1
    else:
        print("game replied:", reply)
        stalled += 1

print("Game Solved:", wins,"against stalled:", stalled)

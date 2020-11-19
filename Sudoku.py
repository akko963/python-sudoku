import inspect
import copy
# from enum import Enum
from dataclasses import dataclass, field
from itertools import chain, permutations, combinations, takewhile
from operator import itemgetter
from random import shuffle
from typing import List, Type, Callable

from sudoku_helper import *

class SudokuData:
    rows: list
    cols: list
    blocks: list

    def __init__(self, slots: list=None):
        self.rows = list([0 for x in range(9)] for y in range(9))
        self.cols = list([0 for x in range(9)] for y in range(9))
        self.blocks = list([0 for x in range(9)] for y in range(9))
        if not slots:
            print("Empty SudokuData")
            return
        for x in range(9):
            for y in range(9):
                a, b = ref[(x, y)][2]
                self.rows[x][y] = self.cols[y][x] = self.blocks[a][b] = slots[x][y]

    def update(self, slots):
        if not check_sudoku(slots):
            print("Error creating Sudoku. Wrong Input.")
        for x in range(9):
            for y in range(9):
                a, b = ref[(x, y)][2]
                self.rows[x][y] = self.cols[y][x] = self.blocks[a][b] = slots[x][y]

    def check_entry(self, x, y, z):
        a, b = ref[(x, y)][2]
        if self.rows[x][y] != 0 or not 0 < z <= 9 or \
                z in self.rows[x] or z in self.cols[y] or z in self.blocks[a][b]:  # Redundant check
            return False
        return True

    def check_entries(self, entries: List):
        return all(self.check_entry(x, y, z) for (x, y, z) in entries)


class Game:
    def __init__(self, slots: List=None):
        self.starter = SudokuData(slots=slots)
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
    def load_file(file_name):
        list_of_lists = []
        print(file_name)
        with open(file_name) as f:
            for i, line in enumerate(f):
                if i == 9:
                    break
                line = line[:-2]
                inner_list = [int(elt.strip()) for elt in line.split(' ')]
                # in alternative, if you need to use the file content as numbers
                # inner_list = [int(elt.strip()) for elt in line.split(',')]
                list_of_lists.append(inner_list)

        return list_of_lists[0:9]

    def load(self, file_name):
        list_of_lists = Game.load_file(file_name)
        self.starter.update(list_of_lists)
        return list_of_lists

    def get_sudoku(self):
        return self.games[-1]


class Trial:
    def __init__(self, slots, tryout_list=None):
        self.channels = []
        self.queue = []
        self.trials = []
        self.play_records = []
        self.add_trial(slots)
        self.log = []
        self.original = SudokuData(slots)

    def add_trial(self, slots):
        print("Added new trial")
        self.trials.append(SudokuData(slots))
        self.channels = [*self.trials[-1].rows, *self.trials[-1].cols, *self.trials[-1].blocks]
        self.trials[-1].play_records = []
        return len(self.trials) - 1

    def check_trial_sudoku(self, trial: int=0):
        my_board = self.trials[trial]
        if not self.check_original(trial):
            return False
        if not check_sudoku(my_board.rows):
            print("\nERROR: Invalid Sudoku: Trial No.", trial, "\nInvalid entries on my_board.\n")
            return False
        if not check_sudoku(convert_to_original(my_board.blocks)) or not check_sudoku(convert_rotate(my_board.cols)):
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
        print(type(slots), slots, len(slots), len(plays))
        assert type(slots[2]) == list and slots and slots[2] and len(slots[2]) == len(plays)
        # for idx, slot in enumerate(slots):
        #     assert 0 <= slot[0] < 9 and 0 <= slot[0] < 9 and 1 <= plays[idx] <= 9
        slot_pairs = translate_list(*slots)
        return all(self.update_entry((x, y, plays[i]), trial) for i, (x, y) in enumerate(slot_pairs))

    def update_entries(self, entries, trial: int=0):
        return all(self.update_entry(entry, trial) for entry in entries)

    def update_entry(self,entry, trial: int=0):
        x, y, z = entry
        slots = self.trials[trial]
        if slots.rows[x][y] and slots.rows[x][y] != z:
            print("Error: Previously filled:", (x, y, slots.rows[x][y]), "new", entry, self.trials[trial].play_records)
            assert False
        elif slots.rows[x][y] == z:
            print("Duplicate Entry")
            return True
        bx, by = ref[(x, y)][2]
        if z in slots.rows[x] or z in slots.cols[y] or z in slots.blocks[bx]:
            print("Error: update_entry. collision with others" ,self.trials[trial].play_records)
            assert False
        # print("Debug: Adding entry:", entry)
        self.trials[trial].play_records.append(((x, y, z),inspect.stack()[1][3]))

        slots.cols[y][x] = slots.rows[x][y] = z
        slots.blocks[(x // 3) * 3 + y // 3][(y % 3) + (x % 3 * 3)] = z
        return True

    def check_solved(self, trial=0):
        sudoku_check, rate = self.check_trial_sudoku(trial), check_progress(self.trials[trial].rows)
        if sudoku_check and rate == 81:
            rate = check_progress(self.trials[trial].rows)
            print("Sudoku is solved. #",  len(self.trials[trial].play_records), "# Block filled =", rate, (rate / 81) * 10000 // 100, "%")
            print_sudoku(self.trials[trial].rows)
            return True
        elif sudoku_check:
            print("\nUpdate #", len(self.trials[trial].play_records), "# Block filled =", rate, (rate / 81) * 10000 // 100, "%")
            print_sudoku(self.trials[trial].rows)
        else:
            print_sudoku(self.trials[trial].rows)
            print("Failed Sudoku check\n")
        return False

    def auto_solve(self, trial=0):
        channel_check = "update"
        if self.check_solved(trial):
            return 81
        while channel_check == "update":
            channel_check = self.try_channels(trial)
            if channel_check == "solved" or self.check_solved(trial):
                print("* ** Solved! ** *")
                print_sudoku(self.trials[trial].rows)
                return 81
            elif channel_check != "update":
                if check_progress(self.trials[trial].rows) == 81:
                    print_sudoku(self.trials[trial].rows)
                    assert self.check_solved(trial)
                    return 81
                print("Error, other than 'update' or 'solved'. Auto-solve functions return:", channel_check)
                print("final message:", channel_check)
                return check_progress(self.trials[trial].rows), len(self.trials[trial].play_records)

    def try_channels(self, trial=0):
        # Approach #1 - explore trials for each block (row/col)
        # list, set, set
        slot_counts, full_slots, pairs_set = self.try_easy(trial)
        if len(full_slots) == 27:
            return "solved"
        indices_set = set(range(27))
        three_plus = indices_set - full_slots - pairs_set
        new_channels = []
        for slot_idx in three_plus:
            plays = list(play_set - set(self.channels[slot_idx]) - {0})
            if len(play_set) < 3:
                print(plays, slot_idx)
                print_sudoku(self.trials[trial].rows)
                assert False
            new_channels.append((slot_counts[slot_idx], slot_idx, plays))
        # contents: count, index (row/col/block + its idx) , plays [1-9]
        new_channels.sort()
        total = len(new_channels)
        half = total // (1+ total//20 + total//14)
        top = new_channels[:half]
        bottom = new_channels[half:]
        packages = self.uniq_channels(top, trial)
        if packages is None:
            print("top half")
            return "update"
        if half != total:
            packages2 = self.uniq_channels(bottom, trial)
            if packages2 is None:
                return "update"
            else:
                print(packages)
                print(packages2)
                print(pairs_set)
                print("No progress made. Bottom half ran without progress.")
                return "no_progress"
            # bottom half does not need to run if top-half == all
        else:    # None above are true: halved and update
            print("List is shortened")
            return "no_progress"
        # Game end here

    def try_easy(self, trial=0):  # Fill those with empty-count: 1 (or) 2 if possible (after confirmation)
        stop, fill_counts = 9*3, [0 for _ in range(9*3)]
        slots = self.channels
        while True:
            idx, full_slots, pair_slots = 0, set(), set()
            while idx < stop:
                if idx in full_slots:
                    idx += 1
                    continue
                elif idx in pair_slots:
                    if len(play_set - set(slots[idx])) <= 1:
                        pair_slots.discard(idx)
                fill_counts[idx] = len(play_set - set(slots[idx]))
                if fill_counts[idx] == 0:
                    full_slots.add(idx)
                    idx += 1    # back to inner while-loop: idx++ incremented
                    continue
                elif fill_counts[idx] == 1:
                    source, my_idx, my_slot = table[idx//9], idx % 9, slots[idx].index(0)
                    my_play = 45 - sum(slots[idx])
                    x, y = translate(source, my_idx, my_slot)
                    if not self.update_entry((x,y, my_play)):
                        print("Error: try-easy-update: Shouldn't happen. Make sure it's not happening")
                        assert False
                    full_slots.add(idx)
                elif fill_counts[idx] == 2:  # if either one (not both) is unplayable, we can fill these
                    indices = [ix for ix, x in enumerate(slots[idx]) if x == 0]
                    plays = play_set - set(slots[idx])
                    assert len(indices) == 2 == len(plays)   # Just to make sure we get it right
                    # Unpack these values
                    source, my_idx = table[idx//9], idx % 9
                    slot_1, slot_2 = indices[0] % 9, indices[1] % 9
                    q1, q2 = ref[(source, my_idx, slot_1)][3], ref[(source, my_idx, slot_2)][3]
                    # more unpacking, get the other lists (row,col,block) of the play's
                    q1_1, q1_2 = table2[q1[0][0]] * 9 + q1[0][1], table2[q1[1][0]] * 9 + q1[1][1]
                    q2_1, q2_2 = table2[q2[0][0]] * 9 + q2[0][1], table2[q2[1][0]] * 9 + q2[1][1]
                    s1_1, s1_2 = slots[q1_1], slots[q1_2]
                    s2_1, s2_2 = slots[q2_1], slots[q2_2]
                    t1 = (play_set - set(s1_1 + s1_2)) & plays
                    t2 = (play_set - set(s2_1 + s2_2)) & plays
                    if t1 == t2:
                        pair_slots.add(idx)
                        self.log.append(((source, my_idx, slot_1), (source, my_idx, slot_1), plays, "Branch try_easy"))
                    else:
                        playable = remove_dupes([t1, t2], [slot_1,slot_2])
                        if playable:
                            for play in playable:
                                _slot, _play = play
                                self.update_entry((*translate(source,my_idx,_slot), _play))
                            break
                        else:
                            pair_slots.add(idx)
                            self.log.append(((source, my_idx, slot_1),(source, my_idx, slot_1), plays, "Branch try_easy"))

                idx += 1    # back to inner while-loop: idx++ incremented
            if idx >= stop:
                return fill_counts, full_slots, pair_slots
            else:
                # print("forever loop")
                pass  # outer-loop ends. idx++ prevented. Recalculate fill count.

    def uniq_channels(self, new_channels, trial=0):
        packages = []
        for channel in new_channels:
            if not channel[2]:
                continue
            package = self.uniq_chan(channel)
            if not package:
                return None
            packages.append(package)
        return packages

    def uniq_chan(self,channel, trial=0):
        _, idx, picks = channel   # picks =  # 3, 4, 5
        source, my_idx = table[idx // 9], idx % 9
        slots = [ix for ix, x in enumerate(self.channels[idx]) if x == 0]
        amount = len(picks)
        if not len(slots) == amount:
            print("Error: parsed data invalid in uniq_chan", slots, amount, picks, idx, self.channels[idx], self.trials[trial].cols[1])
            assert False
        other_sets = []
        other_frees = []
        reduced_sets = []
        debug_set = []
        for slot in slots:
            # more unpacking, get the other lists (row,col,block) of the play's - save in list of sets
            q1, q2 = ref[(source, my_idx, slot)][3]  # others
            q1_idx, q2_idx = table2[q1[0]] * 9 + q1[1],  table2[q2[0]] * 9 + q2[1]
            occupied_others = set(self.channels[q1_idx] + self.channels[q2_idx]) - {0}
            other_frees.append(play_set - occupied_others)
            reduced_sets.append((play_set - occupied_others) & set(picks))
            debug_set.append(((q1,q1_idx),(source,my_idx,slot),(q2,q2_idx)))
        for sdx in range(amount):
            if len(reduced_sets[sdx]) == 1:
                new_play = reduced_sets[sdx].pop()
                entry = *translate(source, my_idx, slots[sdx]), new_play
                self.update_entry(entry)
                if new_play not in picks:
                    print("Error: added/updated using others-check", (source, my_idx, slots[sdx]) ,other_sets[sdx], picks, new_play)
                    print("debug set",debug_set[sdx])
                    assert False
                return None
        reduced_lists = [sorted(rs) for rs in reduced_sets]
        r_sums = [len(rs) for rs in reduced_sets]

        if amount <= 2:
            assert False  # because try_easy took care of them
        reduced_lists.sort()
        reduced_lists.sort(key=len)
        self.log.append(("log",amount-1,reduced_lists)) # debug
        if len(reduced_lists[0]) == 2 and reduced_lists[1] == reduced_lists[2]:
            rl2 = reduced_lists[2:]
            test_list = [rx for irx, rx in enumerate(rl2) if set(reduced_lists[0]) <= set(rx)]
            if len(test_list) == 1:
                if len(test_list[0]) == 3:
                    marked = set(test_list[0]) - set(reduced_lists[0])
                    marked = marked.pop()
                    marked_i = reduced_sets.index(set(test_list[0]))
                    entry = *translate(source, my_idx, slots[marked_i]), marked

                    self.update_entry(entry)
                    return None
        elif len(reduced_lists[0]) == 2 and reduced_lists[1] <= reduced_lists[2] :
            pass

        sh_plays = permutations(picks, len(picks))
        good_plays = []
        for ip, play in enumerate(sh_plays):
            si = 0
            while si < amount:
                if play[si] not in other_frees[si]:
                    break
                si += 1
            if si == amount:
                good_plays.append(play)
        if not good_plays:
            print("Error: No good_plays found?? Bad Sudoku possible.", )
            assert False
        if len(good_plays) == 1:
            for si in range(amount):
                entry = *translate(source, my_idx, slots[si]), good_plays[0][si]
                self.update_entry(entry)
            return None
        elif len(good_plays) > 1:
            updated = False
            for si in range(amount):
                if all(good_plays[0][si] == g_play for g_play in good_plays):
                    entry = *translate(source, my_idx, slots[si]), good_plays[0][si]
                    self.update_entry(entry)
                    updated = True
            if updated:
                return None
        return ["package"]

    def make_channel(self, idx, trial=0):
        source, my_idx = table[idx // 9], idx % 9
        slots = [ix for ix, x in enumerate(self.channels[idx]) if x == 0]
        picks = play_set - set(slots) - {0}
        slot_count = len(picks)

        if not len(slots) == slot_count:
            print("Error: parsed data invalid in uniq_chan", slots, slot_count, picks, idx, self.channels[idx],
                  self.trials[trial].cols[1])
            assert False
        other_frees = []
        reduced_sets = []
        debug_set = []

        r_sums = []
        for slot in slots:
            # more unpacking, get the other lists (row,col,block) of the play's - save in list of sets
            q1, q2 = ref[(source, my_idx, slot)][3]  # others
            q1_idx, q2_idx = table2[q1[0]] * 9 + q1[1], table2[q2[0]] * 9 + q2[1]
            occupied_others = set(self.channels[q1_idx] + self.channels[q2_idx]) - {0}
            other_frees.append(play_set - occupied_others)
            reduced_sets.append((play_set - occupied_others) & set(picks))
            debug_set.append(((q1, q1_idx), (source, my_idx, slot), (q2, q2_idx)))
        return (slot_count, my_idx, slots, picks)
        # remove debug block -> indent backward

class Channel:
    slot_count = 0
    slot_index = 0
    slots = []
    good_plays = []

    def __init__(self, sudoku, index):
        stop, fill_counts = 9*3, [0 for _ in range(9*3)]

def remove_dupes(lists_plays, slot_list):
    slot_count = len(slot_list)
    assert len(lists_plays) == slot_count

    playable = []
    rplist = copy.deepcopy(lists_plays)
    rpcopy = copy.deepcopy(lists_plays)

    def remove_ones():
        nonlocal rplist, rpcopy, playable
        while any(_play for _play in rpcopy if len(_play) == 1):
            rm_list = [_play for _play in rpcopy if len(_play) == 1]
            for found in rm_list:
                ix = rplist.index(found)
                (item,) = found
                playable.append((slot_list[ix], item))
                rplist[ix] = set()
                rpcopy[rpcopy.index(found)] = set()
                for _i in range(slot_count):
                    if found <= rpcopy[_i]:
                        rpcopy[_i] = rpcopy[_i] - found
                    if found <= rplist[_i]:
                        rplist[_i] = rplist[_i] - found

    def remove_set(rmset):
        nonlocal rplist, rpcopy, playable
        for _i in range(slot_count):
            if rmset <= rpcopy[_i]:
                rpcopy[_i] = rpcopy[_i] - rmset
            if rmset <= rplist[_i]:
                rplist[_i] = rplist[_i] - rmset

    remove_ones()
    low_count = 0
    has_dupes = []
    while low_count < slot_count and len(rpcopy[low_count]) < 2:
        low_count += 1
    if low_count >= len(rpcopy) - 1:  # Nothing left on the list
        return playable
    else:
        for rdy in range(len(rpcopy)-low_count-1):
            _dupes = [y for y in rpcopy[rdy+1:] if rpcopy[rdy] <= y]
            if _dupes:
                has_dupes.append((rpcopy[rdy], _dupes))
        for _k in range(len(has_dupes)-1):
            if any(_r for _r in has_dupes[_k+1:] if _r[0] == has_dupes[_k][0]):
                has_dupes[_k] = tuple()
        while tuple() in has_dupes:
            has_dupes.remove(tuple())
        if has_dupes:
            marked_set = set()
            while has_dupes:
                def marking_set(marked):
                    nonlocal has_dupes, marked_set
                    marked_set |= marked
                    for _j in range(len(has_dupes[1:])):
                        has_dupes[1+_j][0] -= marked_set
                        _dupe1 = has_dupes[1+_j][1]
                        for _x in range(len(_dupe1)):
                            _dupe1[_x] -= marked_set
                org, dupes = has_dupes.pop(0)
                pot = copy.deepcopy(org)
                if len(dupes[0]) == 1:
                    recreated = pot |  dupes[0]
                    p_index = rplist.index(recreated)
                    (play,) = dupes[0]
                    playable.append((slot_list[p_index], play))
                    marking_set(dupes[0])
                    removable_list = [dupe - org for dupe in dupes[1:]]
                    for removable in removable_list:
                        if len(removable) == 1:
                            recreated = pot | removable
                            p_index = rplist.index(recreated)
                            (play,) = removable
                            playable.append((slot_list[p_index], play))
                            marking_set(removable)
                    print("nested removal of dupes, removing dupes from processed has_dupes", playable, marked_set, removable_list )
                    continue
                for _i in range(len(dupes)):
                    pot |= dupes[_i]
                    if 2 + _i == len(pot) and dupes[_i+1:]:
                        marking_set(pot)
                        removable_list = [dupe - pot for dupe in dupes[_i:]]
                        for removable in removable_list:
                            if len(removable) == 1:
                                recreated = pot | removable
                                p_index = rplist.index(recreated)
                                (play,) = removable
                                playable.append((slot_list[p_index], play))
                                marking_set(removable)
            print("End of remove_dupes:", playable, marked_set, "from", lists_plays, slot_list )
            return playable
        return[]

def load_file(file_string):
    list_of_lists = []
    print(file_string)
    with open(file_string) as f:
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
# file = "s01b.txt"


loaded = load_file("filelist")
look = generate_lookup()
# printList([(k,look[k] )for k in look.keys() if k[0]=='cols'])
# k = ('cols',3,7)
# print(k,look[k])

wins, stalled = 0, 0
games_progresses = []
for k in range(0, 7):
    file_str = Game.load_file(loaded[k])
    my_game = Game(file_str)
    reply = my_game.auto_solve()
    if reply == 81:
        wins += 1
    else:
        print("game replied:", reply)
        stalled += 1
        games_progresses.append(reply)
    print("Game Solved:", wins, "against stalled:", stalled)

print()
print("Game Solved:", wins, "against stalled:", stalled)
print("Unsolved games progresses made:", games_progresses)


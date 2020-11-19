def print_sudoku_ascii(slots):
    stick, bridge = '\u2502', '\u2501'
    for i, s in enumerate(slots):
        start, mid, mid2, end = ('\u2523', '\u253f', '\u254b', '\u252b') if i else (
            '\u250f', '\u252f', '\u2533', '\u2513')
        print('\u2520' + (('\u2500' * 3 + '\u253c') * 2 + ('\u2500' * 3 + '\u2542')) * 2 + (
                '\u2500' * 3 + '\u253c') * 2 + '\u2500' * 3 + '\u2528' if i % 3 else start + (
                (bridge * 3 + mid) * 2 + (bridge * 3 + mid2)) * 2 + (bridge * 3 + mid) * 2 + bridge * 3 + end)
        print(
            f"\u2503 {s[0]} \u2502 {s[1]} \u2502 {s[2]} \u2503 {s[3]} \u2502 {s[4]} \u2502 {s[5]} \u2503 {s[6]} \u2502 {s[7]} \u2502 {s[8]} \u2503 ".replace(
                '0', ' '))
    else:
        start, mid, mid2, end, bridge = ('\u2517', '\u2537', '\u253b', '\u251b', '\u2501')
        print(
            start + ((bridge * 3 + mid) * 2 + (bridge * 3 + mid2)) * 2 + (bridge * 3 + mid) * 2 + bridge * 3 + end)
ref=[]


from dataclasses import dataclass
@dataclass
class SudokuData:
    rows: list = None

    def __post_init__(self):
        self.cols = list([0 for i in range(9)] for j in range(9))
        self.blocks = list([0 for i in range(9)] for j in range(9))
        if not self.rows:
            self.rows = list([0 for i in range(9)] for j in range(9))
            return
        for i in range(9):
            for j in range(9):
                self.blocks[(i // 3) * 3 + j // 3][(j % 3) + (i % 3 * 3)] = self.cols[j][i] = self.rows[i][j]

# Scratch #save #aug26
        # print("Check emptiness(gen):", any(self.trials[trial].rows[x][y]
        #                                    for ch in all_ch for (x, y) in translate_list(ch[1], ch[2], ch[0])))
        # print("Check non-empty(gen)", all(self.trials[trial].rows[x][y]
        #                                   for ch in all_ch for (x, y) in translate_list_inverse(ch[1], ch[2], ch[0])))
        # print("Check = matches slots")
        # print("True = all() matches", all(self.trials[trial].rows[x][y] == self.trials[trial].ref[ch[0]][ch[1]] [translate(x,y,to=ch[0])[1]]
        #                                   for ch in all_ch for (x, y) in translate_list_inverse(ch[1], ch[2], ch[0])))
        # printList((self.trials[trial].rows[x][y] , self.trials[trial].ref[ch[0]][ch[1]],ch[0], (x,y),translate(x,y,to=ch[0]), self.trials[trial].ref[ch[0]][ch[1]] [translate(x,y,to=ch[0])[1]] )
        #           for ch in all_ch for (x, y) in translate_list_inverse(ch[1], ch[2], ch[0]))
    keys = []
    def print_board(board=keys, start=0):
        print()
        if board is not None and start >=81:  # board is key check
            temp = board
            board = [(item[1], item[2]) for item in temp[start:start+81]]
            start = 0
        def tl(pair): return f'{pair[0]}{pair[1]}'
        def bl(block): return " ".join(tl(j) for j in block)

        for i in range(9):
            k = i*9
            print(bl(board[start + k: start + k + 9]))


    def box_check(self, source, index, y, bid, trial=0):
        func = {'rows': self.row_check, 'cols': self.col_check, 'blocks': self.block_check}
        return func[source](index, y, bid, trial)

    def box_checks(self, source, index, y_list, bids, trial=0):
        func = {'rows': self.row_check, 'cols': self.col_check, 'blocks': self.block_check}
        if not len(y_list)== len(bids):
            print("\n\n#$  ERROR   ")
            print(source,index,y_list, bids)

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

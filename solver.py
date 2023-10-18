import numpy as np
import copy
class Solve_Main:
    def __init__(self,board):

        self.board = np.array(board).reshape(-1)
        self.is_default = np.where(self.board!=0,True,False).reshape(-1)

        self.old_span =  [ [ _ for _ in range(1,10)] for i in range(81)]
        self.init_old_span()
    def init_old_span(self):
        for index,value in enumerate(self.board):
            if value!=0:
                    self.old_span = self.add_new_value(value,index,self.old_span)

    def solve(self):



        return self.help_solve(self.board,0,self.old_span),self.board
    def help_solve(self,board,index,span):

        if index >= 81:
            self.board = board
            return True


        if self.is_default[index]:
            return self.help_solve(board,index+1,span)

        for value in span[index]:

            new_board = board.copy()
            new_span = self.add_new_value(value,index,span)
            new_board[index] = value

            if self.help_solve(new_board,index+1,new_span) :
                return True


        return False
    def add_new_value(self,value,index,span):

        new_span = copy.deepcopy(span)
        i  = index//9
        j =  index-i*9

        i_nho = i-i%3
        j_nho = j-j%3


        for col in range(9):
            if col==j or value not in new_span[i*9+col]:
                continue
            new_span[i*9+col].remove(value)

        for row in range(9):
            if row==i or value not in new_span[row*9+j]:
                continue

            new_span[row*9+j].remove(value)

        for row in range(3):
            for col in range(3):

                if (row==i_nho and col == j_nho) or value not  in new_span[(i_nho+row)*9+(j_nho+col)]:
                    continue
                new_span[(i_nho+row)*9+(j_nho+col)].remove(value)



        return  new_span

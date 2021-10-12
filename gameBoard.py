# import numpy as np
# a = np.array([[1,2, 4, 4], [5, 5, 3,4]])
# a = a.flatten()

# a = a.reshape(2, 4)

# data = np.asarray([0, 1, 2, 0])

# random_action = np.unravel_index(np.argmax(data, axis=None), data.shape)[0]

# print(random_action)


# # print(a)
import tkinter as tk
class GameBoard(tk.Frame):
    def __init__(self, parent, rows=10,walls = [], columns=10, size=48, color1="white", color2="white"):
        '''size is the size of a square, in pixels'''

        self.rows = rows
        self.columns = columns
        self.size = size
        self.color1 = color1
        self.color2 = color2
        self.walls = walls
        self.pieces = {}
        canvas_width = columns * size
        canvas_height = rows * size

        tk.Frame.__init__(self, parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=canvas_width, height=canvas_height, background="bisque")
        self.canvas.pack(side="top", fill="both", expand=True, padx=2, pady=2)

        # this binding will cause a refresh if the user interactively
        # changes the window size
        self.canvas.bind("<Configure>", self.refresh)

    def addpiece(self, name, image, row=0, column=0):
        '''Add a piece to the playing board'''
        self.canvas.create_image(0,0, image=image, tags=(name, "piece"), anchor="nw")
        self.placepiece(name, row, column)

    def placepiece(self, name, row, column):
        '''Place a piece at the given row/column'''
        self.pieces[name] = (row, column)
        x0 = (column * self.size) + int(self.size/2)
        y0 = (row * self.size) + int(self.size/2)
        self.canvas.coords(name, x0, y0)

    def refresh(self, event):
        '''Redraw the board, possibly in response to window being resized'''
        xsize = int((event.width-1) / self.columns)
        ysize = int((event.height-1) / self.rows)
        self.size = min(xsize, ysize)
        self.canvas.delete("square")
        color = self.color2
    
        
        for row in range(self.rows):
            color = self.color1 if color == self.color2 else self.color2


            for col in range(self.columns):
                x1 = (col * self.size)
                y1 = (row * self.size)
                x2 = x1 + self.size
                y2 = y1 + self.size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=color, tags="square")
                color = self.color1 if color == self.color2 else self.color2


        if(len(self.walls) > 0):

            for wall in self.walls:
                x1 = (wall[1] * self.size)
                y1 = (wall[0] * self.size)
                x2 = x1 + self.size
                y2 = y1 + self.size

                print(wall[0])
                print(wall[1])
                # raise Exception
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill="grey", tags="square")
                color = self.color1 if color == self.color2 else self.color2



        for name in self.pieces:
            self.placepiece(name, self.pieces[name][0], self.pieces[name][1])
        self.canvas.tag_raise("piece")
        self.canvas.tag_lower("square")


# image comes from the silk icon set which is under a Creative Commons
# license. For more information see http://www.famfamfam.com/lab/icons/silk/
# imagedata = '''
#     R0lGODlhEAAQAOeSAKx7Fqx8F61/G62CILCJKriIHM+HALKNMNCIANKKANOMALuRK7WOVLWPV9eR
#     ANiSANuXAN2ZAN6aAN+bAOCcAOKeANCjKOShANKnK+imAOyrAN6qSNaxPfCwAOKyJOKyJvKyANW0
#     R/S1APW2APW3APa4APe5APm7APm8APq8AO28Ke29LO2/LO2/L+7BM+7BNO6+Re7CMu7BOe7DNPHA
#     P+/FOO/FO+jGS+/FQO/GO/DHPOjBdfDIPPDJQPDISPDKQPDKRPDIUPHLQ/HLRerMV/HMR/LNSOvH
#     fvLOS/rNP/LPTvLOVe/LdfPRUfPRU/PSU/LPaPPTVPPUVfTUVvLPe/LScPTWWfTXW/TXXPTXX/XY
#     Xu/SkvXZYPfVdfXaY/TYcfXaZPXaZvbWfvTYe/XbbvHWl/bdaPbeavvadffea/bebvffbfbdfPvb
#     e/fgb/Pam/fgcvfgePTbnfbcl/bfivfjdvfjePbemfjelPXeoPjkePbfmvffnvbfofjlgffjkvfh
#     nvjio/nnhvfjovjmlvzlmvrmpvrrmfzpp/zqq/vqr/zssvvvp/vvqfvvuPvvuvvwvfzzwP//////
#     ////////////////////////////////////////////////////////////////////////////
#     ////////////////////////////////////////////////////////////////////////////
#     ////////////////////////////////////////////////////////////////////////////
#     ////////////////////////////////////////////////////////////////////////////
#     ////////////////////////////////////////////////////////////////////////////
#     /////////////////////////////////////////////////////yH+FUNyZWF0ZWQgd2l0aCBU
#     aGUgR0lNUAAh+QQBCgD/ACwAAAAAEAAQAAAIzAD/CRxIsKDBfydMlBhxcGAKNIkgPTLUpcPBJIUa
#     +VEThswfPDQKokB0yE4aMFiiOPnCJ8PAE20Y6VnTQMsUBkWAjKFyQaCJRYLcmOFipYmRHzV89Kkg
#     kESkOme8XHmCREiOGC/2TBAowhGcAyGkKBnCwwKAFnciCAShKA4RAhyK9MAQwIMMOQ8EdhBDKMuN
#     BQMEFPigAsoRBQM1BGLjRIiOGSxWBCmToCCMOXSW2HCBo8qWDQcvMMkzCNCbHQga/qMgAYIDBQZU
#     yxYYEAA7
# '''



# if __name__ == "__main__":
#     root = tk.Tk()
#     walls = [[0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [10, 6], [9, 6], [8, 6], [7, 6], [6, 6], [5, 6], [4, 6]]

#     board = GameBoard(root, walls = walls)
#     board.pack(side="top", fill="both", expand="true", padx=4, pady=4)
#     player1 = tk.PhotoImage(data=imagedata)
#     board.addpiece("player1", player1, 0,0)
#     root.mainloop()
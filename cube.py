# R - 0
#  L, R
#  1, 2
#
# O - 1 
#  L, R
#  3, 4
#
# Y - 2
#  L, R
#  5, 6
#
# G - 3
#  L, R
#  7, 8
#
# B - 4
#  L, R 
#  9, 10
#
# W - 5
#  L, R
#  11, 12
import numpy as np

def getRandomMove(prev=None):
    if prev is None:
        return np.random.randint(1, 13)
    else:
        rand = np.random.randint(1, 13)
        while rand == prev:
            rand = np.random.randint(1, 13)
        return rand
def getAllMoves(prev=None):
    moves = []
    for i in range(12):
        i += 1
        if i is not prev:
            moves.append(i)
    return moves
    
def getInverseMove(move):
    if move == 1:
        return 2
    elif move == 2:
        return 1
    elif move == 3:
        return 4
    elif move == 4:
        return 3
    elif move == 5:
        return 6
    elif move == 6:
        return 5
    elif move == 7:
        return 8
    elif move == 8:
        return 7
    elif move == 9:
        return 10
    elif move == 10:
        return 9
    elif move == 11:
        return 12
    elif move == 12:
        return 11
    elif move == 0:
        return 0
    else:
        raise Exception("Invalid move: ", move)

class Cube:
    
    # looking green,blue,white,yellow,then red is on top; when looking at red,orange, then white is on top
    def __init__(self):
        self.prevMove = None
        self.prevState = None
        self.reversedMove = False # for training, punish if cube returns to previous state
        #0 Red
        self.red = np.full((3,3), 0)
        #1 Orange
        self.orange = np.full((3,3), 1)
        #2 Yellow
        self.yellow = np.full((3,3), 2)
        #3 Green
        self.green = np.full((3,3), 3)
        #4 Blue
        self.blue = np.full((3,3), 4)
        #5 White
        self.white = np.full((3,3), 5)

    def getSolvedState(self):
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        
    def wasReversed(self):
        return self.prevMove
    
    def howCloseToSolved(self, state):
        return np.sum(state == self.getSolvedState())
    
    def closerToSolved(self):
        return self.howCloseToSolved(self.getStateArray()) > self.howCloseToSolved(self.prevState)
        
    def printNicely(self):
        block = np.block([self.red, self.orange, self.yellow, self.green, self.blue, self.white])
        print(block)
        return block
    
    def isSolved(self):
        return np.all(self.getStateArray() == self.getSolvedState())
    
    def reset(self):
        self.red = np.full((3,3), 0)
        self.orange = np.full((3,3), 1)
        self.yellow = np.full((3,3), 2)
        self.green = np.full((3,3), 3)
        self.blue = np.full((3,3), 4)
        self.white = np.full((3,3), 5)
        self.prevMove = None
        self.reversedMove = False

    def randomize(self):
        prev = None
        for _ in range(20):
            move = getRandomMove(prev)
            prev = move
            self.move(move)

    def getStateArray(self):
        # Loop through faces and compose an array for each, then extend into one array
        state = []
        for face in [self.red, self.orange, self.yellow, self.green, self.blue, self.white]:
            state.extend(int(x) for x in face.flatten())  # Ensure elements are cast to Python's int type
        return state

    
    def getStateMatrix(self):
        return np.array([self.red, self.orange, self.yellow, self.green, self.blue, self.white])


    # HARD CODED ROTATIONS:
    def RL(self): # 1
        self.red = np.rot90(self.red, k=1)
        temp = self.green[0,:].copy()
        self.green[0,:] = self.white[0,:].copy()
        self.white[0,:] = self.blue[0,:].copy()
        self.blue[0,:] = self.yellow[0,:].copy()
        self.yellow[0,:] = temp
    def RR(self): # 2
        self.red = np.rot90(self.red, k=-1)
        temp = self.green[0,:].copy()
        self.green[0,:] = self.yellow[0,:].copy()
        self.yellow[0,:] = self.blue[0,:].copy()
        self.blue[0,:] = self.white[0,:].copy()
        self.white[0,:] = temp
    def OL(self): # 3
        self.red = np.rot90(self.red, k=1)
        temp = self.green[2,:].copy()
        self.green[2,:] = self.yellow[2,:].copy()
        self.yellow[2,:] = self.blue[2,:].copy()
        self.blue[2,:] = self.white[2,:].copy()
        self.white[2,:] = temp
    def OR(self): # 4
        self.red = np.rot90(self.red, k=-1)
        temp = self.green[2,:].copy()
        self.green[2,:] = self.white[2,:].copy()
        self.white[2,:] = self.blue[2,:].copy()
        self.blue[2,:] = self.yellow[2,:].copy()
        self.yellow[2,:] = temp
    def YL(self): # 5
        self.yellow = np.rot90(self.yellow, k=1)
        temp = self.green[:,2].copy()
        self.green[:,2] = self.red[2,:].copy()
        self.red[2,:] = self.blue[:,0].copy()
        self.blue[:,0] = self.orange[2,:].copy()
        self.orange[2,:] = temp
    def YR(self): # 6
        self.yellow = np.rot90(self.yellow, k=-1)
        temp = self.blue[:,0].copy()
        self.blue[:,0] = self.red[2,:].copy()
        self.red[2,:] = self.green[:,2].copy()
        self.green[:,2] = self.orange[2,:].copy()
        self.orange[2,:] = temp
    def GL(self): # 7
        self.green = np.rot90(self.green, k=1)
        temp = self.white[:,2].copy()
        self.white[:,2] = self.red[:,0].copy()
        self.red[:,0] = self.yellow[:,0].copy()
        self.yellow[:,0] = self.orange[:,2].copy()
        self.orange[:,2] = temp
    def GR(self): # 8
        self.green = np.rot90(self.green, k=-1)
        temp = self.white[:,2].copy()
        self.white[:,2] = self.orange[:,2].copy()
        self.orange[:,2] = self.yellow[:,0].copy()
        self.yellow[:,0] = self.red[:,0].copy()
        self.red[:,0] = temp
    def BL(self): # 9
        self.blue = np.rot90(self.blue, k=1)
        temp = self.red[:,2].copy()
        self.red[:,2] = self.white[:,0].copy()
        self.white[:,0] = self.orange[:,0].copy()
        self.orange[:,0] = self.yellow[:,2].copy()
        self.yellow[:,2] = temp
    def BR(self): # 10
        self.blue = np.rot90(self.blue, k=-1)
        temp = self.red[:,2].copy()
        self.red[:,2] = self.yellow[:,2].copy()
        self.yellow[:,2] = self.orange[:,0].copy()
        self.orange[:,0] = self.white[:,0].copy()
        self.white[:,0] = temp
    def WL(self): # 11
        self.white = np.rot90(self.white, k=1)
        temp = self.green[:,0].copy()
        self.green[:,0] = self.orange[0,:].copy()
        self.orange[0,:] = self.blue[:,2].copy()
        self.blue[:,2] = self.red[0,:].copy()
        self.red[0,:] = temp
    def WR(self): # 12
        self.white = np.rot90(self.white, k=-1)
        temp = self.green[:,0].copy()
        self.green[:,0] = self.red[0,:].copy()
        self.red[0,:] = self.blue[:,2].copy()
        self.blue[:,2] = self.orange[0,:].copy()
        self.orange[0,:] = temp

    # expects number 1-12, 
    def move(self, move):
        # move -= 1
        if move < 0 or move > 13:
            raise Exception("Invalid move: ", move)
        self.prevState = self.getStateArray()
        if move == 0 or move == 13:
            return
        elif move == 1:
            self.RL()
        elif move == 2:
            self.RR()
        elif move == 3:
            self.OL()
        elif move == 4:
            self.OR()
        elif move == 5:
            self.YL()
        elif move == 6:
            self.YR()
        elif move == 7:
            self.GL()
        elif move == 8:
            self.GR()
        elif move == 9:
            self.BL()
        elif move == 10:
            self.BR()
        elif move == 11:
            self.WL()
        elif move == 12:
            self.WR()
        if self.prevMove == getInverseMove(move):
            self.reversedMove = True
        else:
            self.reversedMove = False
        self.prevMove = move

    def runGUI(self):
        import tkinter as tk
        from tkinter import ttk

        # Define a mapping from face values to colors
        color_map = {
            0: 'red',    # R
            1: 'orange', # O
            2: 'yellow', # Y
            3: 'green',  # G
            4: 'blue',   # B
            5: 'white'   # W
        }

        def update_faces():
            for i, face in enumerate([self.red, self.orange, self.yellow, self.green, self.blue, self.white]):
                for row in range(3):
                    for col in range(3):
                        value = face[row, col]
                        face_labels[i][row][col].config(text=str(value), background=color_map[value])

        def create_face_grid(frame):
            return [[ttk.Label(frame, text='0', borderwidth=1, relief="solid", width=4) for _ in range(3)] for _ in range(3)]

        root = tk.Tk()
        root.title("Rubik's Cube")

        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        face_labels = []
        face_names = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'White']
        for i, face_name in enumerate(face_names):
            frame = ttk.LabelFrame(main_frame, text=face_name, padding="5")
            frame.grid(row=i//3, column=i%3, padx=5, pady=5)
            face_grid = create_face_grid(frame)
            for row in range(3):
                for col in range(3):
                    face_grid[row][col].grid(row=row, column=col)
            face_labels.append(face_grid)

        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.grid(row=2, column=0, columnspan=3)

        buttons = [
            ('Red Left', self.RL), ('Red Right', self.RR),
            ('Orange Left', self.OL), ('Orange Right', self.OR),
            ('Yellow Left', self.YL), ('Yellow Right', self.YR),
            ('Green Left', self.GL), ('Green Right', self.GR),
            ('Blue Left', self.BL), ('Blue Right', self.BR),
            ('White Left', self.WL), ('White Right', self.WR),
            ('Reset', self.reset), ('Get State', self.getStateArray)
        ]

        for i, (text, command) in enumerate(buttons):
            button = ttk.Button(button_frame, text=text, command=lambda cmd=command: [cmd(), update_faces()])
            button.grid(row=i//2, column=i%2, padx=5, pady=5)

        update_faces()
        root.mainloop()

if __name__ == "__main__":
    cube = Cube()
    cube.runGUI()




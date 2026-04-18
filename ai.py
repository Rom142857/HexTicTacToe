from random import *
def bot(current_player,board,remaining):
    scores={}
    for player in (-current_player,current_player):
      for i in board:
        if board[i]!=player:
          continue
        for j in ((1,0,0),(0,1,1),(1,-1,2)):
          for k in range(5):
            pos=list(i)
            pos[0]-=j[0]*k
            pos[1]-=j[1]*k
            candidates=[]
            n=0
            for l in range(6):
              if tuple(pos) in board:
                if board[tuple(pos)]!=player:
                  candidates=[]
                  break
                else:
                  n+=1
              else:
                candidates.append(tuple(pos))
              pos[0]+=j[0]
              pos[1]+=j[1]
            m=0
            if n==5:
              if player==current_player:
                  m=100000000000000
              else:
                  m=10000000000
            if n==4:
              if player==current_player:
                if remaining==2:
                  m=1000000000000
                else:
                  m=0
              else:
                  m=100000000
            if n==3:
              if player==current_player:
                  m=1000000
              else:
                  m=100
            if n==2:
              if player==current_player:
                  m=10000
              else:
                  m=1
            if m!=0:
              for candidate in candidates:
                if candidate not in scores:
                  scores[candidate]=m
                else:
                  scores[candidate]+=m
            else:
              if candidates!=[]:
                null_move=choice(candidates)
    if scores=={}:
      return null_move
    best_move=max(map(lambda x:scores[x],list(scores)))
    moves=list(filter(lambda x:scores[x]==best_move,list(scores)))
    return choice(moves)
class MyBot(Bot):
    """"""

    def get_move(self, game):
        if not game.board:
            return (0, 0)

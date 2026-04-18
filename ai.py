class MyBot(Bot):
    """"""

    def get_move(self, game):
        if not game.board:
            return (0, 0)

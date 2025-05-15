from aisuite.framework.message import Message


class Choice:
    def __init__(self):
        self.message = Message()

    @staticmethod
    def create_choice(m: Message):
        c = Choice()
        c.message = m
        return c

from abc import abstractmethod


class SplitBase(object):
    def __init__(self, split_config):
        self.split_config = split_config

    @abstractmethod
    def split(self):
        pass


class SplitIID(SplitBase):
    def split(self):
        pass


class SplitNonIID(SplitBase):
    def split(self):
        pass

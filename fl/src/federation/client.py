from flwr.client import NumPyClient


class FLClient(NumPyClient):
    def __init__(self) -> None:
        super().__init__()
from requests.models import Response


class RequestResponse(Response):
    def __init__(self, status_code: int, content: str):
        super().__init__()
        self.status_code = status_code
        self._content = content.encode("utf-8")  # must be bytes

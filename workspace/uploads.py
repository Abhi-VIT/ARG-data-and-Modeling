from django.conf import settings
from django.core.files.uploadhandler import FileUploadHandler, StopUpload


class CappedUploadHandler(FileUploadHandler):
    """Stop multipart bodies before writing an unbounded file to disk."""
    def __init__(self, request=None):
        super().__init__(request)
        self.received = 0

    def receive_data_chunk(self, raw_data, start):
        self.received += len(raw_data)
        if self.received > settings.MAX_UPLOAD_BYTES:
            self.request.upload_too_large = True
            raise StopUpload(connection_reset=False)
        return raw_data

    def file_complete(self, file_size):
        return None

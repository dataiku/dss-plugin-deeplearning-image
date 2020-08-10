import os


class VirtualManagedFolder:
    def __init__(self, path):
        self.path = path

    def get_absolute_path(self, *args):
        return os.path.join(self.path, *args)

    def get_download_stream(self, file_path):
        return open(self.get_absolute_path(file_path), 'rb')

    def get_path_details(self, path):
        absolute_path = self.get_absolute_path(path)
        return {
            'exists': os.path.exists(absolute_path),
            'directory': os.path.isdir(absolute_path)
        }

    def list_paths_in_partition(self):
        return os.listdir(self.path)

    def upload_stream(self, path, f):
        raise IOError("VirtualManagedFolder do not have upload_stream property.")

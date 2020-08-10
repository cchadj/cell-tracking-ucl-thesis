from sharedvariables import *

class CellSelectorModel:
    video_session: VideoSession

    def __init__(self):
        '''
        Initializes the two members the class holds:
        the file name and its contents.
        '''
        self.video_session = None
        self.fileName = None
        self.fileContent = ""

    def create_video_session(self, filename):
        try:
            f = open(filename, 'r')
            f.close()
            self.video_session = VideoSession(filename)
            return self.video_session
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename}' does not exist or is not readable")



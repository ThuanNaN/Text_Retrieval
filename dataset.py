import os
from utils.data_preprocessing import preprocessing

class DataIterator:
    def __init__(self, data):
        self._data = data
        self._idx = 0

    def __next__(self):
        if self._idx < len(self._data):
            doc_id = self._data[self._idx]['doc_id']
            content = self._data[self._idx]['content']
            self._idx += 1
            return {
                'doc_id': doc_id,
                'content': content
            }
        raise StopIteration

class Dataset:
    """
        Create `Dataset` object from document folder.
    """
    def __init__(self, doc_root, isPreprocessing=True) -> None:

        doc_filepaths = sorted(os.listdir(doc_root), key=lambda path:int(path.split('.')[0]))
        self.docs = []
     
        for f in doc_filepaths:
            obj = {}
            obj['doc_id'] = int(f.split(".")[0])
            path_file = os.path.join(os.getcwd(),doc_root.split("/")[-1],f)

            with open(path_file) as f_read:
                content = " ".join(line.rstrip() for line in f_read.readlines())
                if isPreprocessing:
                    content = preprocessing(content)

                obj['content'] = content
            self.docs.append(obj)


    def __len__(self):
        return len(self.docs)
    
    def __getitem__(self, idx):
        return self.docs[idx]

    def __iter__(self):
        return DataIterator(self.docs)
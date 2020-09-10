import numpy


class QTS:
    '''
    this class is to construct datasets
    the file qts.txt contains 10,000 tang peoms
    '''
    def __init__(self, path):
        self.dictionary = {}
        self.chars = []
        self.poems = []
        lines = None
        with open(path, encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip()
            self.calculate_chars(line)
        self.num_examples = len(lines)
        self.pos = numpy.random.randint(0, self.num_examples)
        self.train = self

    def next_batch(self, batch_size):
        next = self.pos + batch_size
        result = None
        if next < self.num_examples:
            result = self.poems[self.pos: next]
        else:
            result = self.poems[self.pos:]
            next -= self.num_examples
            result.extend(self.poems[: next])
        self.pos = next
        return [result]

    def calculate_chars(self, line):
        poem = []
        for char in line:
            if char not in self.dictionary:
                id = len(self.chars)
                self.dictionary[char] = id
                self.chars.append(char)
            poem.append(self.dictionary[char])
        self.poems.append(poem)

    def get_chars(self, *ids):
        result = [self.chars[id] for id in ids]
        return ''.join(result)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('Data destroyed.')


if __name__ == '__main__':
    qts = QTS('/home/qiangde/Downloads/qts.txt')
    a = qts.next_batch(10)
    for sentence in a:
        print(qts.get_chars(*sentence))
def chain(*iterators):
    for iterator in iterators:
        for elem in iterator:
            yield elem 
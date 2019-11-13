import numpy as np

class BucketingParallelDataLoader:

    def __init__(self, dataloader, n=20, length_fn=lambda seq: len(seq.split()),add_reverse=False,add_tags=False):
        """
        Sorts by source sentence length descending (nice for RNN encoders),
        then by target length.
        """
        self.dataloader = dataloader
        self.length_fn = length_fn
        self.it = iter(dataloader)
        self.n = n
        self.add_reverse=add_reverse
        self.add_tags=add_tags

        self._sort_next_batches()
        self.batch_size = dataloader.batch_size


    def _sort_next_batches(self):
        count = 0
        src_batches = []
        tgt_batches = []
        src_rev_batches= []
        tgt_rev_batches= []
        src_shuf_batches= []
        tgt_shuf_batches= []
        for batch in self.it:
            #TODO. change depending on value of add_tags
            if self.add_reverse:
                src_batch, tgt_batch, src_rev_batch, tgt_rev_batch, src_shuf_batch, tgt_shuf_batch = batch
            else:
                src_batch, tgt_batch = batch
            src_batches += src_batch
            tgt_batches += tgt_batch
            if self.add_reverse:
                src_rev_batches+=src_rev_batch
                tgt_rev_batches+=tgt_rev_batch
                src_shuf_batches+=src_shuf_batch
                tgt_shuf_batches+=tgt_shuf_batch
            count += 1
            if count == self.n:
                break

        if len(src_batches) == 0:
            self.it = iter(self.dataloader)
            raise StopIteration

        sort_keys = sorted(range(len(src_batches)),
                            #key=lambda idx: (len(src_batches[idx].split()),
                            #                len(tgt_batches[idx].split())),
                            key=lambda idx: (self.length_fn(src_batches[idx]),
                                            self.length_fn(tgt_batches[idx])),
                            reverse=True)
        src_batches = np.array(src_batches)
        tgt_batches = np.array(tgt_batches)

        self.sorted_src_batches = src_batches[sort_keys]
        self.sorted_tgt_batches = tgt_batches[sort_keys]

        if self.add_reverse:
            src_rev_batches = np.array(src_rev_batches)
            tgt_rev_batches = np.array(tgt_rev_batches)

            self.sorted_src_rev_batches = src_rev_batches[sort_keys]
            self.sorted_tgt_rev_batches = tgt_rev_batches[sort_keys]

            src_shuf_batches = np.array(src_shuf_batches)
            tgt_shuf_batches = np.array(tgt_shuf_batches)

            self.sorted_src_shuf_batches = src_shuf_batches[sort_keys]
            self.sorted_tgt_shuf_batches = tgt_shuf_batches[sort_keys]

        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.sorted_src_batches):
            self._sort_next_batches()
        start_idx = self.idx
        end_idx = self.idx + self.batch_size
        self.idx += self.batch_size
        if self.add_reverse:
            return (self.sorted_src_batches[start_idx:end_idx],
                self.sorted_tgt_batches[start_idx:end_idx],self.sorted_src_rev_batches[start_idx:end_idx],self.sorted_tgt_rev_batches[start_idx:end_idx],self.sorted_src_shuf_batches[start_idx:end_idx],self.sorted_tgt_shuf_batches[start_idx:end_idx])
        else:
            return (self.sorted_src_batches[start_idx:end_idx],
                self.sorted_tgt_batches[start_idx:end_idx])

class BucketingTextDataLoader:

    def __init__(self, dataloader, n=20, length_fn=lambda seq: len(seq.split())):
        """
        Sorts by sentence length descending (nice for RNN encoders).
        """
        self.dataloader = dataloader
        self.length_fn = length_fn
        self.it = iter(dataloader)
        self.n = n
        self._sort_next_batches()
        self.batch_size = dataloader.batch_size

    def _sort_next_batches(self):
        count = 0
        batches = []
        for batch in self.it:
            batches += batch
            count += 1
            if count == self.n:
                break

        if len(batches) == 0:
            self.it = iter(self.dataloader)
            raise StopIteration

        sort_keys = sorted(range(len(batches)),
                            #key=lambda idx: len(batches[idx].split()),
                            key=lambda idx: self.length_fn(batches[idx]),
                            reverse=True)
        batches = np.array(batches)

        self.sorted_batches = batches[sort_keys]
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.sorted_batches):
            self._sort_next_batches()
        start_idx = self.idx
        end_idx = self.idx + self.batch_size
        self.idx += self.batch_size
        return self.sorted_batches[start_idx:end_idx]

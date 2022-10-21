from typing import List, Tuple, Union

import torch


class Beam(object):
    def __init__(self, size: int, sos: torch.Tensor, eos: List[Union[int, List[int]]]):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prev_ks: List[torch.Tensor] = []
        # The outputs at each time-step.
        self.next_ys: List[torch.Tensor] = [self.tt.LongTensor(size).fill_(0)]
        self.next_ys[0][:] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished: List[Tuple[torch.Tensor, int, int]] = []

    def get_current_state(self) -> torch.LongTensor:
        """ Get the outputs for the current timestep. """
        batch = self.tt.LongTensor(self.next_ys[-1]).view(-1, 1)
        return batch

    def get_current_origin(self) -> torch.Tensor:
        """ Get the backpointers for the current timestep. """
        return self.prev_ks[-1]

    def advance(self, word_lk: torch.Tensor) -> None:
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Args:
            word_lk: probs of advancing from the last step (K x words)
        """

        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)

            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] in self._eos:
                    beam_lk[i] = -1e20
        else:
            beam_lk = word_lk[0]
        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] in self._eos:
                self.finished.append((self.scores[i], len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] in self._eos:
            self.eosTop = True

    def done(self):
        """ Return whether the beam search has been finished. """
        return self.eosTop and len(self.finished) >= self.size

    def get_final(self) -> List[Tuple[torch.Tensor, int, int]]:
        """ Get the final finished tuple. """
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.next_ys) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])

        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] not in self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.next_ys) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def get_hypothesis(self, beam_res: List[Tuple[torch.Tensor, int, int]]) \
            -> List[List[torch.Tensor]]:
        """ Walk back to construct the full hypothesis. """
        hypothesis: List[List[torch.Tensor]] = []
        for _, timestep, k in beam_res:
            hyp: List[torch.Tensor] = []
            for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
                hyp.append(self.next_ys[j + 1][k])
                k = self.prev_ks[j][k]
            hypothesis.append(hyp[::-1])
        return hypothesis
    
    def build_target_tokens(self, predictions: List[List[torch.Tensor]]) \
            -> List[List[List[torch.Tensor]]]:
        """ Build the target."""
        sentence = []
        for prediction in predictions:
            tokens = []
            for tok in prediction:
                tokens.append(tok)
                if tok in self._eos:
                    break
            sentence.append(tokens)
        return sentence

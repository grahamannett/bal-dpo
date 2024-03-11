class Trainer:
    def bpo_loss(self, batch_inputs):
        sample1 = self.model.generate(**batch_inputs)
        sample2 = self.model.generate(**batch_inputs)
        highscore, lowscore = self.compare_samples(sample1, sample2)

    def compare_samples(self, sample1, sample2):
        scores = [(self.model.score(sample), sample) for sample in [sample1, sample2]]
        scores.sort(key=lambda x: x[0])
        return scores

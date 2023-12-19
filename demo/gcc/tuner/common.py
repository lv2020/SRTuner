# Define constant
FLOAT_MAX = float('inf')
import pickle
import time

class FlagInfo:
    def __init__(self, name, configs):
        self.name = name
        self.configs = configs

class Evaluator:
    def __init__(self, path, num_repeats):
        self.path = path
        self.num_repeats = num_repeats
    
    def build(self):
        assert 0, "Undefined"

    def run(self):
        assert 0, "Undefined"

    def evaluate(self):
        assert 0, "Undefined"

    def clean(self):
        assert 0, "Undefined"


class Tuner:
    def __init__(self, search_space, evaluator, args = None, name = "Base Tuner", default_setting = None):
        self.search_space = search_space
        self.evaluator = evaluator
        self.args = args
        self.name = name
        self.default_setting = default_setting
        self.default_perf = evaluator.evaluate(default_setting)[-1]
        self.visited = set()

        print(f"default_perf : {self.default_perf:.3f}")

    
    def generate_candidates(self, batch_size=1):
        assert 0, "Undefined"
    
    def evaluate_candidates(self, candidates):
        assert 0, "Undefined"

    def reflect_feedback(perfs):
        assert 0, "Undefined"

    def tune(self, budget, batch_size=1):
        best_opt_setting, best_perf = None, FLOAT_MAX
        i = 0
        op_his = []
        begin = time.time()
        while 1:
            if self.args.time_limitation:
                with open(f'{self.args.env}_{self.args.time_limitation}_SRTuner_{self.args.random_seed}.pkl', 'wb') as f:
                    pickle.dump(op_his, f)
                if time.time() - begin > self.args.time_limitation:
                    break
            else:
                with open(f'{self.args.env}_{self.args.steps}_SRTuner_{self.args.random_seed}.pkl', 'wb') as f:
                    pickle.dump(op_his, f)
                if i > self.args.steps:
                    break

            candidates = self.generate_candidates(batch_size=batch_size)
            res = self.evaluate_candidates(candidates)[0]
            res.append(time.time() - begin)
            perfs = [res[-2]]
            op_his.append(res)
            i += len(candidates)
            for opt_setting, perf in zip(candidates, perfs):
                if perf < best_perf:
                    best_perf = perf
                    best_opt_setting = opt_setting
            
            
            print(f"[{i}] current trial: {perf:.3f}s, best performance so far: {best_perf:.3f}s")
            
            self.reflect_feedback(perfs)
        return best_opt_setting, best_perf
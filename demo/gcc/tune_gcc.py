import os, subprocess, re

from tuner import FlagInfo, Evaluator, FLOAT_MAX
from tuner import RandomTuner, SRTuner
import pickle
import json
import argparse
import time
import numpy as np
# Define GCC flags
class GCCFlagInfo(FlagInfo):
    def __init__(self, name, configs, isParametric, stdOptLv):
        super().__init__(name, configs)
        self.isParametric = isParametric
        self.stdOptLv = stdOptLv


# Read the list of gcc optimizations that follows certain format.
# Due to a slight difference in GCC distributions, the supported flags are confirmed by using -fverbose-asm.
# Each chunk specifies flags supported under each standard optimization levels.
# Besides flags identified by -fverbose-asm, we also considered flags in online doc.
# They are placed as the last chunk and considered as last optimization level.
# (any standard optimization level would not configure them.)
def read_gcc_opts(path):
    search_space = dict() # pair: flag, configs
    # special case handling
    search_space["stdOptLv"] = GCCFlagInfo(name="stdOptLv", configs=[1,2,3], isParametric=True, stdOptLv=-1)
    with open(path, "r") as fp:
        stdOptLv = 0
        for raw_line in fp.read().split('\n'):
            # Process current chunk
            if(len(raw_line)):
                line = raw_line.replace(" ", "").strip()
                if line[0] != '#':
                    tokens = line.split("=")
                    flag_name = tokens[0]
                    # Binary flag
                    if len(tokens) == 1:
                        info = GCCFlagInfo(name=flag_name, configs=[False, True], isParametric=False, stdOptLv=stdOptLv)
                    # Parametric flag
                    else:
                        assert(len(tokens) == 2)
                        info = GCCFlagInfo(name=flag_name, configs=tokens[1].split(','), isParametric=True, stdOptLv=stdOptLv)
                    search_space[flag_name] = info
            # Move onto next chunk
            else:
                stdOptLv = stdOptLv+1
    return search_space


def convert_to_str(opt_setting, search_space):
    str_opt_setting = " -O" + str(opt_setting["stdOptLv"])

    for flag_name, config in opt_setting.items():
        assert flag_name in search_space
        flag_info = search_space[flag_name]
        # Parametric flag
        if flag_info.isParametric:
            if flag_info.name != "stdOptLv" and len(config)>0:
                str_opt_setting += f" {flag_name}={config}"
        # Binary flag
        else:
            assert(isinstance(config, bool))
            if config:
                str_opt_setting += f" {flag_name}"
            else:
                negated_flag_name = flag_name.replace("-f", "-fno-", 1)
                str_opt_setting += f" {negated_flag_name}"
    return str_opt_setting


# Define tuning task
class cBenchEvaluator(Evaluator):
    def __init__(self, path, num_repeats, search_space, artifact="a.out"):
        super().__init__(path, num_repeats)
        self.artifact = artifact
        self.search_space = search_space
        self.compile_config = json.load(open('new_config.f'))

    def build(self, str_opt_setting):
        '''
        commands = f"""cd {self.path};
        make clean > /dev/null 2>/dev/null;
        make -j4 CCC_OPTS_ADD="{str_opt_setting}" LD_OPTS=" -o {self.artifact} -fopenmp" > /dev/null 2>/dev/null;
        """
        subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True).wait()

        # Check if build fails
        if not os.path.exists(self.path + "/" + self.artifact):
            return -1
        return 0
        '''
        config = self.compile_config
        op_seq = str_opt_setting
        load_lib = ''
        if os.path.exists('libfunc.so'):
            load_lib = '-L. -lfunc'
        s = time.time()
        m = os.popen(f'rm -f {config["exe_file"]}; rm -f *.o').read()
        for f in config['files']:
            #for c++
            if f.endswith('.cpp') or f.endswith('.cc'):
                CC = 'g++ -w '
            else:
                CC = 'gcc -w '
            m = os.popen(f'{CC} {op_seq} {config["lib"]} -c {f} -o {f.replace("/", "_").replace("..", "").split(".")[0]}.o {load_lib} > tmp').read()
        m = os.popen(f'{CC} {op_seq} *.o -o {config["exe_file"]} {config["link_lib"]} {load_lib} > tmp').read()
        if not os.path.exists(config['exe_file']):
            return -1
        return 0

    def get_timing_result(self):
        cwd = os.getcwd()
        #run spec programs
        if 'spec' in cwd:
            pid = cwd.split('/')[-1]
            os.chdir(f'../../spec_programs/{pid}')
            os.popen(f'cp ../../spec_benchmarks/{pid}/{self.compile_config["exe_file"]} ./').read()
            s = time.time()
            for cmd in self.compile_config['run']:
                process = subprocess.Popen(f'./{self.compile_config["exe_file"]} {cmd}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell = True)
                try:
                    return_code = process.wait(timeout=1000)
                except:
                    os.chdir(cwd)
                    return FLOAT_MAX
            e = time.time() - s
            os.popen(f'rm -f {self.compile_config["exe_file"]}').read()
            os.chdir(cwd)
            if return_code != 0:
                return FLOAT_MAX
            return e
        #for MiBench and PolyBench
        else:
            res = []
            for i in range(10):
                process = subprocess.Popen(f'LD_LIBRARY_PATH=. ./{self.compile_config["exe_file"]} {self.compile_config["run"]}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell = True)
                try:
                    output, error = process.communicate(timeout = 60)
                except:
                    return FLOAT_MAX
                tmp = [i.strip() for i in str(error)[:-1].strip().split('\n') if 'timing' in i]
                tmp = [i.split('timing:')[-1].strip()[:-2] for i in tmp]
                if len(tmp) == 0:
                    return FLOAT_MAX
                else:
                    t = []
                    for i in tmp:
                        i = i.split(' ')
                        try:
                            t.append(int(i[1].strip()) - int(i[0].strip()))
                        except:
                            continue
                res.append(sum(t))
            return np.median(res)

    def run(self, num_repeats, input_id=1):
        '''
        run_commands = f"""cd {self.path};
        ./_ccc_check_output.clean ;
        ./__run {input_id} 2>&1;
        """
        verify_commands = f"""cd {self.path};
        rm -f tmp-ccc-diff;
        ./_ccc_check_output.diff {input_id};
        """
        tot = 0

        # Repeat the measurement and get the averaged execution time
        for _ in range(num_repeats):
            # Run the executable
            p = subprocess.Popen(run_commands, stdout=subprocess.PIPE, shell=True)
            p.wait()
            stdouts = p.stdout.read().decode('ascii').split("\n")

            # Check if the output is correct
            subprocess.Popen(verify_commands, stdout=subprocess.PIPE, shell=True).wait()
            diff_file = self.path+ "/tmp-ccc-diff"
            if os.path.isfile(diff_file) and os.path.getsize(diff_file) == 0:
                # Runs correctly. Extract performance numbers.
                for out in stdouts:
                    if out.startswith("real"):
                        out = out.replace("real\t", "")
                        nums = re.findall("\d*\.?\d+", out)
                        assert len(nums) == 2, "Expect %dm %ds format"
                        secs = float(nums[0])*60+float(nums[1])
                        tot += secs
            else:
                # Runtime error or wrong output
                return FLOAT_MAX

        # Correct execution
        return tot/num_repeats
        '''
        return self.get_timing_result()


    def evaluate(self, opt_setting, num_repeats=-1):
        flags = convert_to_str(opt_setting, self.search_space)
        compile_begin = time.time()
        error = self.build(flags)
        compile_time = time.time() - compile_begin
        if error == -1:
            # Bulid error
            return [flags, FLOAT_MAX, FLOAT_MAX]

        # If not specified, use the default number of repeats
        if num_repeats == -1:
            num_repeats = self.num_repeats

        perf = self.run(num_repeats, input_id=2)
        #self.clean()

        #return perf
        return [flags, compile_time, perf]


    def clean(self):
        commands = f"""cd {self.path};
        make clean > /dev/null 2>/dev/null;
        ./_ccc_check_output.clean ;
        """
        subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True).wait()


if __name__ == "__main__":
    '''
    # Assign the number of trials as the budget.
    budget = 1000
    # Benchmark info
    benchmark_home = "./cBench"
    benchmark_list = ["network_dijkstra", "consumer_jpeg_c", "telecom_adpcm_d"]
    gcc_optimization_info = "gcc_opts.txt"

    # Extract GCC search space
    search_space = read_gcc_opts(gcc_optimization_info)
    default_setting = {"stdOptLv":3}

    with open("tuning_result.txt", "w") as ofp:
        ofp.write("=== Result ===\n")
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='gcc')
    parser.add_argument('--run_dir', type=str, default='')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--steps', type = int, default = 120)
    parser.add_argument('--time_limitation', type = int, default = None)
    parser.add_argument('--algo', type = str, default = 'srtuner')
    args = parser.parse_args()

    '''
    for benchmark in benchmark_list:
        path = benchmark_home + "/" + benchmark + "/src"
        evaluator = cBenchEvaluator(path, num_repeats=30, search_space=search_space)
        
        tuners = [
            RandomTuner(search_space, evaluator, default_setting),
            SRTuner(search_space, evaluator, default_setting)
        ]

        for tuner in tuners:
            best_opt_setting, best_perf = tuner.tune(budget)
            if best_opt_setting is not None:
                default_perf = tuner.default_perf
                best_perf = evaluator.evaluate(best_opt_setting)
                print(f"Tuning {benchmark} w/ {tuner.name}: {default_perf:.3f}/{best_perf:.3f} = {default_perf/best_perf:.3f}x")
                with open("tuning_result.txt", "a") as ofp:
                    ofp.write(f"Tuning {benchmark} w/ {tuner.name}: {default_perf:.3f}/{best_perf:.3f} = {default_perf/best_perf:.3f}x\n")
    '''
    gcc_optimization_info = "gcc_opts.txt"
    search_space = read_gcc_opts(gcc_optimization_info)
    default_setting = {"stdOptLv":3}

    os.chdir(args.run_dir)
    evaluator = cBenchEvaluator('./', num_repeats=30, search_space=search_space)
    if args.algo == 'srtuner':
        tuner = SRTuner(search_space, evaluator, args, default_setting)
    else:
        tuner = RandomTuner(search_space, evaluator, args, default_setting)
    best_opt_setting, best_perf = tuner.tune(0)


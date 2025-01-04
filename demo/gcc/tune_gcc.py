import os, subprocess, re, sys

from tuner import FlagInfo, Evaluator, FLOAT_MAX
from tuner import RandomTuner, SRTuner
import pickle
import json
import argparse
import time
import numpy as np
if 'x' in os.popen('hostname').read():
    project_path = os.path.abspath('/mnt/scratch/e/e0509838/Project/RL_tuner/gym_compiler/envs')
else:
    project_path = os.path.abspath('/home/liwei/Project/RL_method/gym_compiler/envs')
sys.path.insert(0, project_path)
from compiler_env import *

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
    search_space["stdOptLv"] = GCCFlagInfo(name="stdOptLv", configs=[1], isParametric=True, stdOptLv=-1)
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
            '''
            else:
                negated_flag_name = flag_name.replace("-f", "-fno-", 1)
                str_opt_setting += f" {negated_flag_name}"
            '''
    return str_opt_setting


# Define tuning task
class cBenchEvaluator(Evaluator):
    def __init__(self, env, path, num_repeats, search_space, artifact="a.out"):
        super().__init__(path, num_repeats)
        self.env = env
        self.artifact = artifact
        self.search_space = search_space
        #self.compile_config = json.load(open(f'{args.run_dir}/new_config.f'))

    def build(self, str_opt_setting):
        if args.env == 'gcc':
            op_seq = str_opt_setting
            src_folder = None
            if 'spec' in args.run_dir.lower():
                src_folder = args.run_dir
            #clean and build executable file
            load_lib = ''
            if os.path.exists('libfunc.so'):
                load_lib = '-L. -lfunc'
            s = time.time()
            config = self.compile_config
            m = os.popen(f'rm -f {config["exe_file"]}; rm -f *.o').read()
            for f in config['files']:
                #for c++
                if f.endswith('.cpp') or f.endswith('.cc'):
                    CC = 'g++ -w '
                else:
                    CC = 'gcc -w '
                if src_folder:
                    m = os.popen(f'{CC} {op_seq} -I{src_folder}/ {config["lib"].replace("-I", f"-I{src_folder}/")} -c {src_folder}/{f} -o {f.replace("/", "_").replace("..", "").split(".")[0]}.o {load_lib} > tmp').read()
                else:
                    m = os.popen(f'{CC} {op_seq} {config["lib"]} -c {f} -o {f.replace("/", "_").replace("..", "").split(".")[0]}.o {load_lib} > tmp').read()
            if src_folder:
                m = os.popen(f'{CC} {op_seq} *.o -o {config["exe_file"]} -I{src_folder}/ {config["link_lib"].replace("-I", f"-I{src_folder}")} {load_lib} > tmp').read()
            else:
                m = os.popen(f'{CC} {op_seq} *.o -o {config["exe_file"]} {config["link_lib"]} {load_lib} > tmp').read()
            compile_time = time.time() - s
            return 0
        else:
            op_seq = str_opt_setting
            src_folder = None
            if 'spec' in args.run_dir.lower():
                src_folder = args.run_dir
            load_lib = ''
            if os.path.exists('libfunc.so'):
                load_lib = '-L. -lfunc'
            #clean and build executable file with gcc
            s = time.time()
            config = self.compile_config
            m = os.popen(f'rm -f {config["exe_file"]}; rm -f *.bc; rm -f *.o').read()
            for f in config['files']:
                #for c++
                if f.endswith('.cpp') or f.endswith('.cc'):
                    CC = '/home/e/e0509838/Project/llvm/bin/clang++ -w -stdlib=libc++'
                else:
                    CC = '/home/e/e0509838/Project/llvm/bin/clang -w '
                base_filename = f.replace("/", "_").replace("..", "").split('.')[0]
                if src_folder:
                    m = os.popen(f'{CC} -O0 -I{src_folder}/ {config["lib"].replace("-I", f"-I{src_folder}/")} -emit-llvm -c {src_folder}/{f} -o {base_filename}.bc').read()
                else:
                    m = os.popen(f'{CC} {config["lib"]} -O0 -emit-llvm -c {f} -o {base_filename}.bc {load_lib}').read()
                m = os.popen(f'/home/e/e0509838/Project/llvm/bin/opt -S {op_seq} {base_filename}.bc -o {base_filename}.opt.bc').read()
                m = os.popen(f'/home/e/e0509838/Project/llvm/bin/llc -O0 -filetype=obj {base_filename}.opt.bc').read()
            m = os.popen(f'{CC} *.o -o {config["exe_file"]} {config["link_lib"]} {load_lib}').read()
            return 0

    def get_timing_result(self):
        #run spec programs
        if 'spec' in args.run_dir or 'origin' in args.run_dir:
            if 'spec' in args.run_dir:
                run_count = 1
            else:
                run_count = 5
            res = []
            for i in range(run_count):
                s = time.time()
                for cmd in self.compile_config['run']:
                    process = subprocess.Popen(f'./{self.compile_config["exe_file"]} {cmd}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell = True)
                    try:
                        return_code = process.wait(timeout=1000)
                    except:
                        return FLOAT_MAX
                    if return_code != 0:
                        return FLOAT_MAX
                e = time.time() - s
                res.append(e)
            os.popen(f'rm -f {self.compile_config["exe_file"]}').read()
            return np.median(res)
        #for MiBench and PolyBench
        else:
            res = []
            if 'MediaBench_h264enc' in args.run_dir:
                run_count = 5
            else:
                run_count = 10
            for i in range(run_count):
                if 'liver' in args.run_dir and args.env == 'llvm':
                    process = subprocess.Popen(f'LD_LIBRARY_PATH=/home/e/e0509838/Project/llvm/lib:. ./{self.compile_config["exe_file"]} {self.compile_config["run"]}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell = True)
                else:
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
        compile_time, run_time = self.env.run_single(flags)[0]
        run_time = np.median(run_time)
        if run_time == np.inf:
            return [flags, compile_time, FLOAT_MAX]
        return [flags, compile_time, run_time]


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
    if args.env == 'gcc':
        gcc_optimization_info = "gcc_opts.txt"
    else:
        gcc_optimization_info = "llvm_opts.txt"
    search_space = read_gcc_opts(gcc_optimization_info)
    if 'perlbench' in args.run_dir.lower() or 'mibench_office_rsynth' in args.run_dir.lower():
        default_setting = {"stdOptLv":1}
    else:
        default_setting = {"stdOptLv":3}
    if 'liver' in args.run_dir and '-fpack-struct' in search_space:
        del search_space['-fpack-struct']

    os.chdir(args.run_dir)
    env = CompilerEnv('gcc', [i for i in search_space], args.run_dir, None, None, args.random_seed, args.steps, args.time_limitation)
    evaluator = cBenchEvaluator(env, './', num_repeats=30, search_space=search_space)
    tuner = SRTuner(search_space, evaluator, args, default_setting)
    best_opt_setting, best_perf = tuner.tune(0)

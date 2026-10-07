import re
import sys
file = sys.argv[1]

def find_times(needle, rank):
    time_pat = re.compile(rf'{re.escape(needle)}\s*(\d+(?:\.\d+)?)\s*sec\b')

    times = []

    with open(file, "r", encoding="utf-8") as f:
        for lineno, text in enumerate(f, 1):
            if needle in text and f"[rank={rank}]" in text:
                m = time_pat.search(text)
                # print(text)
                if m:
                    times.append(float(m.group(1)))
    return times

def find_mem(rank):
        
    mem_pat = re.compile(rf'process memory\s*([0-9]+(?:\.[0-9]+)?)\s*GB\b')

    mem_gb = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if f"[rank={rank}]" in line:
                m = mem_pat.search(line)
                if m:
                    mem_gb.append(float(m.group(1)))
    return mem_gb

occurs = {'gradients_cascade:':1,
          'redist:':2,
          'adj_tomo:': 1,
          'gradient_prbfit:':1,
         'allreduce:':2,
         'allreduce2:':2,
         'fwd_tomo:':1,
         'hessian_cascade:':3,
         'hessian_prbfit:':3,
         'linear_batch:':5,
         'linear_redot_batch:':3}
         #'min:':1}

times = {}
rank = 0

for w in occurs.keys():
    times[w] =sum(find_times(w,rank)[-occurs[w]:]) 
mem = find_mem(0)[-1]

total = 0
for w in occurs.keys():
    total+=times[w]
    print(w,times[w])

print(f'total',total)
print(f'mem',mem)

print('no words:')
for w in occurs.keys():
    print(times[w])

print(total)
print(mem)



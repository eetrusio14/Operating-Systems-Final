import threading
import time
import random
import matplotlib.pyplot as plt

# RAM Setup
RAM_SIZE = 2048 # total words in main memory
PAGE_SIZE = 64 # words per page/frame
NUM_PAGES = RAM_SIZE // PAGE_SIZE # total frames available
L1_SIZE = 64 # entries for L1 cache
L2_SIZE = 128 # entries for L2 cache
TIME_SLICE_INSTR = 5 # instructions per time slice for round robin
IO_MIN = 0.05 # minimum simulated I/O delay in seconds
IO_MAX = 0.10 # maximum simulated I/O delay in seconds

# Instruction opcodes
ADD = 0 # addition
SUB = 1 # subtraction
MUL = 2 # multiplication
DIV = 3 # integer division
LOAD = 4 # load from memory to accumulator
STORE = 5 # store accumulator to memory
AND = 6 # bitwise AND
OR = 7 # bitwise OR
JUMP = 8 # unconditional jump
JZ = 9 # jump if zero
SYSCALL = 10 # system call
IO = 11 # simulated I/O operation
NOP = -1 # no operation

# Represents the system performance metrics for Module 5 analysis
class SimMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.cpu_active_ticks = 0
        self.total_ticks = 0
        self.context_switches = 0
        self.completed_processes = []
        self.gantt_data = []

    # Purpose: Records data when a process finishes execution
    def log_process(self, pcb, current_time):
        turnaround = current_time - pcb.arrival_time
        waiting = turnaround - pcb.burst_executed
        self.completed_processes.append({
            'pid': pcb.pid,
            'burst': pcb.burst_executed,
            'turnaround': turnaround,
            'waiting': max(0, waiting),
            'priority': pcb.priority
        })

    # Purpose: Prints a summary table of the collected metrics
    def print_report(self, algo_name):
        print(f"\nMETRICS REPORT ({algo_name})")
        print(f"{'PID':<5}{'Burst':<10}{'Turnaround':<12}{'Waiting':<10}")
        
        avg_wait, avg_turn = 0, 0
        for p in sorted(self.completed_processes, key=lambda x: x['pid']):
            print(f"{p['pid']:<5}{p['burst']:<10.2f}{p['turnaround']:<12.2f}{p['waiting']:<10.2f}")
            avg_wait += p['waiting']
            avg_turn += p['turnaround']
        
        count = len(self.completed_processes)
        if count > 0:
            avg_wait /= count
            avg_turn /= count
        
        cpu_util = (self.cpu_active_ticks / max(1, self.total_ticks)) * 100
        
        print(f"\nAverage Waiting Time: {avg_wait:.4f}")
        print(f"Average Turnaround Time: {avg_turn:.4f}")
        print(f"CPU Utilization: {cpu_util:.2f}%")
        print(f"Total Context Switches: {self.context_switches}\n")

    # Purpose: Logs Gantt chart data for visualization
    def log_gantt(self, pid, queue_level, start_tick, end_tick):
        if end_tick > start_tick:
            self.gantt_data.append({
                'pid': pid,
                'queue': queue_level,
                'start': start_tick,
                'end': end_tick
            })

    #    # Purpose: Generates a PNG Gantt chart with queue labels (PID X (RQY))
    def export_gantt_chart(self, algo_name):
        if not self.gantt_data:
            return

        # Collect PIDs and map to Y positions
        pids = sorted(list(set(d['pid'] for d in self.gantt_data)))
        pid_to_y = {pid: i for i, pid in enumerate(pids)}

        # Color per queue level (0, 1, 2). Fallback to gray for unknown.
        queue_colors = {
            0: 'tab:blue',   # RQ0
            1: 'tab:orange', # RQ1
            2: 'tab:green'   # RQ2
        }

        fig, ax = plt.subplots(figsize=(10, max(2, len(pids) * 0.8)))

        for pid in pids:
            # All raw slices for this PID
            raw_slices = sorted(
                [d for d in self.gantt_data if d['pid'] == pid],
                key=lambda x: x['start']
            )

            # Merge contiguous slices that have SAME queue and touching times
            merged = []
            if raw_slices:
                cur_q = raw_slices[0]['queue']
                cur_start = raw_slices[0]['start']
                cur_end = raw_slices[0]['end']

                for s in raw_slices[1:]:
                    q = s['queue']
                    st = s['start']
                    en = s['end']

                    # Merge only if contiguous in time *and* same queue level
                    if q == cur_q and st == cur_end:
                        cur_end = en
                    else:
                        merged.append((cur_q, cur_start, cur_end))
                        cur_q, cur_start, cur_end = q, st, en

                # Final pending block
                merged.append((cur_q, cur_start, cur_end))

            y_center = pid_to_y[pid]
            y_bottom = y_center - 0.35
            bar_height = 0.7

            for q, start, end in merged:
                width = end - start
                if width <= 0:
                    continue

                color = queue_colors.get(q, 'gray')

                # Draw bar segment
                ax.broken_barh(
                    [(start, width)],
                    (y_bottom, bar_height),
                    facecolors=color,
                    edgecolor='black'
                )

                # Text label: "P1 (RQ0)" etc, centered in the bar
                label = f"P{pid} (RQ{q})"
                ax.text(
                    start + width / 2.0,
                    y_center,
                    label,
                    ha='center',
                    va='center',
                    fontsize=7
                )

        # Y axis labels
        ax.set_yticks(range(len(pids)))
        ax.set_yticklabels([f"PID {p}" for p in pids])

        ax.set_xlabel("Time (Ticks)")
        ax.set_title(f"Scheduling Gantt Chart: {algo_name}")
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)

        # Legend for queue levels
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color=queue_colors.get(0, 'gray'), label="RQ0 (RR)"),
            Patch(color=queue_colors.get(1, 'gray'), label="RQ1 (HRRN)"),
            Patch(color=queue_colors.get(2, 'gray'), label="RQ2 (FCFS)")
        ]
        ax.legend(handles=legend_handles, title="Queues", loc='upper right')

        filename = f"gantt_{algo_name}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[INFO] Gantt chart saved to '{filename}'")

# Represents a FIFO cache with capacity, replacement, and hit/miss tracking
class FIFOCache:
    def __init__(self, capacity, name):
        self.capacity = capacity
        self.name = name
        self.store = {} # address:value entries
        self.order = [] # maintains FIFO eviction order
        self.hits = 0 # total cache hits
        self.misses = 0 # total cache misses

    # Purpose: Retrieves a value from cache and updates hit/miss counters 
    def get(self, addr):
        if addr in self.store:
            self.hits += 1
            return self.store[addr]
        self.misses += 1
        return None

    # Purpose: Adds or replaces a cache entry using FIFO eviction
    def put(self, addr, value):
        if addr not in self.store:
            if len(self.store) >= self.capacity:
                old = self.order.pop(0)
                if old in self.store:
                    del self.store[old]
            self.order.append(addr)
        self.store[addr] = value

# Represents system main memory (RAM) with read/write operations
class RAM:
    def __init__(self, size):
        self.size = size
        self.mem = [0] * size  # list simulating RAM cells
        self.lock = threading.Lock()  # ensures thread safety

    # Purpose: Reads one word from RAM
    def read(self, addr):
        if addr < 0 or addr >= self.size:
            raise Exception('RAM read out of bounds')
        return self.mem[addr]

    # Purpose: Writes one word to RAM
    def write(self, addr, val):
        if addr < 0 or addr >= self.size:
            raise Exception('RAM write out of bounds')
        self.mem[addr] = val

# Represents a memory block allocated to a process for tracking
class MemoryBlock:
    def __init__(self, pid, start, end, isFree):
        self.processID = pid
        self.start = start
        self.end = end
        self.isFree = isFree

# Represents the system’s memory allocation table 
class MemoryTable:
    def __init__(self):
        self.blocks = [MemoryBlock(-1, 0, RAM_SIZE - 1, 1)]
        self.lock = threading.Lock()
        print("[MEMORY TABLE] Initialized memory table")
        self.print_status()

    # Purpose: Allocates a contiguous block of memory using FIRST_FIT or BEST_FIT
    def allocate(self, pid, size_words, strategy='FIRST_FIT'):
        with self.lock:
            candidate_idx = -1
            best_size = float('inf')

            for i, blk in enumerate(self.blocks):
                if blk.isFree == 1 and (blk.end - blk.start + 1) >= size_words:
                    curr_size = blk.end - blk.start + 1
                    
                    if strategy == 'FIRST_FIT':
                        candidate_idx = i
                        break
                    elif strategy == 'BEST_FIT':
                        if curr_size < best_size:
                            best_size = curr_size
                            candidate_idx = i
            
            if candidate_idx != -1:
                blk = self.blocks[candidate_idx]
                alloc_start = blk.start
                alloc_end = alloc_start + size_words - 1
                
                remaining_start = alloc_end + 1
                blk.start = remaining_start
                
                new_blk = MemoryBlock(pid, alloc_start, alloc_end, 0)
                self.blocks.insert(candidate_idx, new_blk)
                
                if blk.start > blk.end:
                    self.blocks.pop(candidate_idx + 1)
                
                print(f"[MEM] {strategy} Alloc PID {pid} -> {alloc_start}-{alloc_end}")
                return alloc_start, alloc_end

        raise Exception('Out of RAM!')

    # Purpose: Frees all blocks owned by a process and coalesces adjacent free blocks 
    def deallocate(self, pid):
        with self.lock:
            freed_any = False
            for blk in self.blocks:
                if blk.processID == pid:
                    blk.isFree = 1
                    blk.processID = -1
                    freed_any = True
            
            self.blocks.sort(key=lambda b: b.start)
            i = 0
            while i < len(self.blocks) - 1:
                cur, nxt = self.blocks[i], self.blocks[i+1]
                if cur.isFree == 1 and nxt.isFree == 1 and cur.end + 1 == nxt.start:
                    cur.end = nxt.end
                    self.blocks.pop(i+1)
                else:
                    i += 1
            if freed_any:
                print(f"[MEM] Freed PID {pid}")

    # Purpose: Prints the current memory allocation status of each created process
    def print_status(self):
        print("\n[MEMORY TABLE]")
        for blk in self.blocks:
            label = "Free" if blk.isFree == 1 else (f"Used by PID {blk.processID}")
            print(f"{label}: {blk.start} - {blk.end}")

# Represents a page table entry mapping a virtual page to a physical frame
class PageTableEntry:
    def __init__(self, vpn, ppn, valid):
        self.vpn = vpn # virtual page number
        self.ppn = ppn # physical page number
        self.valid = valid # valid bit for mapping status

# Represents the Memory Management Unit (MMU) performing address translation
class MMU:
    def __init__(self, ram, l1, l2):
        self.ram = ram
        self.l1 = l1
        self.l2 = l2
        self.free_frames = list(range(NUM_PAGES))
        self.ptables = {} # page tables per process
        self.pt_lock = threading.Lock()

    # Purpose: Creates new page table entries for a process
    def map_pages(self, pid, num_pages):
        with self.pt_lock:
            if pid not in self.ptables:
                self.ptables[pid] = {}
            pt = self.ptables[pid]
            for _ in range(num_pages):
                if len(self.free_frames) == 0:
                    raise Exception('Out of physical frames')
                ppn = self.free_frames.pop(0)
                vpn = len(pt)
                pt[vpn] = PageTableEntry(vpn, ppn, True)

    # Purpose: Translates a virtual address into a physical address
    def translate(self, pid, vaddr):
        vpn = vaddr // PAGE_SIZE
        offset = vaddr % PAGE_SIZE
        pt = self.ptables.get(pid, {})
        pte = pt.get(vpn)
        if pte is None or not pte.valid:
            raise Exception(f"Page fault for PID {pid} at VPN {vpn}")
        ppn = pte.ppn
        paddr = ppn * PAGE_SIZE + offset
        return paddr

    # Purpose: Reads a value from memory using hierarchical lookup (L1->L2->RAM) 
    def read(self, pid, vaddr):
        paddr = self.translate(pid, vaddr)
        val = self.l1.get(paddr)
        if val is not None:
            return val
        val = self.l2.get(paddr)
        if val is None:
            val = self.ram.read(paddr)
            self.l2.put(paddr, val)
        self.l1.put(paddr, val)
        return val

    # Purpose: Writes a value to memory and updates caches (Write-through)
    def write(self, pid, vaddr, value):
        paddr = self.translate(pid, vaddr)
        self.ram.write(paddr, value)
        self.l1.put(paddr, value) 
        self.l2.put(paddr, value)

# Represents a Process Control Block (PCB) for process tracking
class PCB:
    def __init__(self, pid, burst_estimate=0):
        self.pid = pid
        self.pc = 0 # program counter
        self.acc = 0 # accumulator
        self.ir = (NOP, None) # instruction register
        self.zf = 0 # zero flag
        self.cf = 0 # carry flag
        self.vf = 0 # overflow flag
        self.state = 'NEW' 
        self.queue_level = 0
        
        self.priority = 0  
        self.arrival_time = 0 
        self.burst_time_estimate = burst_estimate 
        self.burst_executed = 0 
        self.start_time = 0

    # Purpose: Calculates Response Ratio for HRRN Algorithm 
    def calculate_hrrn(self, current_time):
        waiting_time = current_time - self.arrival_time - self.burst_executed
        service_time = max(0.1, self.burst_time_estimate)
        return (waiting_time + service_time) / service_time

    # Purpose: Helper to determine remaining time for SRT
    def time_remaining(self):
        return max(0, self.burst_time_estimate - self.burst_executed)

# Represents the OS scheduler with queues and selection logic 
class Scheduler:
    def __init__(self, algorithm='RR'):
        self.ready = [] # ready queue for processes
        self.blocked = [] # blocked queue for I/O
        self.lock = threading.Lock()
        self.algorithm = algorithm 
        self.feedback_queues = {
            0: [], # RQ0 -> RR
            1: [], # RQ1 -> HRRN
            2: [] # RQ2 -> FCFS
            } 

    # Purpose: Adds a process to the ready queue
    def add_ready(self, pcb):
        with self.lock:
            if self.algorithm == 'FEEDBACK':
                pcb.queue_level = 0
                self.feedback_queues[0].append(pcb)
            else:
                if pcb not in self.ready:
                    self.ready.append(pcb)


    # Purpose: Remove a procress from all ready lists/queues (Feeback has multiple)
    def remove_ready(self, pcb):
        with self.lock:
            if self.algorithm != 'FEEDBACK':
                if pcb in self.ready:
                    self.ready.remove(pcb)
                return
            for q in self.feedback_queues.values():
                if pcb in q:
                    q.remove(pcb)

    # Purpose: Adds a process to the blocked queue
    def add_blocked(self, pcb):
        with self.lock:
            if pcb not in self.blocked:
                self.blocked.append(pcb)

    # Purpose: Moves a process from blocked to ready
    def unblock(self, pcb):
        with self.lock:
            if pcb in self.blocked:
                self.blocked.remove(pcb)

            lvl = pcb.queue_level
            if lvl not in self.feedback_queues:
                lvl = 0

            self.feedback_queues[lvl].append(pcb)

    
    # Purpose: Determines which feedback queue a process is in
    def get_queue_level(self, pcb):
        for lvl, q in self.feedback_queues.items():
            if pcb in q:
                return lvl
        return None

    # Purpose: Selects the next process based on the active algorithm
        # Purpose: Selects the next process based on the active algorithm
    def select_next(self, current_time_tick):
        with self.lock:
            if self.algorithm != 'FEEDBACK':

                # Filter out terminated PCBs defensively
                live_ready = [p for p in self.ready if p.state != 'TERMINATED']
                if not live_ready:
                    return None

                if self.algorithm in ['FCFS', 'RR']:
                    # Simple queue head
                    return live_ready[0]

                elif self.algorithm == 'PRIORITY':
                    return max(live_ready, key=lambda p: p.priority)

                elif self.algorithm in ['SRT', 'SPN']:
                    return min(live_ready, key=lambda p: p.time_remaining())

                elif self.algorithm == 'HRRN':
                    return max(
                        live_ready,
                        key=lambda p: p.calculate_hrrn(current_time_tick)
                    )

                # Fallback -> just return head
                return live_ready[0]

            # RQ0 -> RR (just first non-terminated PCB)
            for pcb in self.feedback_queues[0]:
                if pcb.state != 'TERMINATED':
                    return pcb

            # RQ1 -> HRRN among non-terminated
            live_rq1 = [p for p in self.feedback_queues[1] if p.state != 'TERMINATED']
            if live_rq1:
                return max(
                    live_rq1,
                    key=lambda pcb: pcb.calculate_hrrn(current_time_tick)
                )

            # RQ2 -> FCFS (first non-terminated)
            for pcb in self.feedback_queues[2]:
                if pcb.state != 'TERMINATED':
                    return pcb

            return None


    # Purpose: Handles queue rotation and demotion for RR and Feedback algorithms
    def rotate_queue(self, current_pcb):
        with self.lock:
            if self.algorithm != 'FEEDBACK':
                # Round Robin rotation
                if self.algorithm == 'RR' and current_pcb in self.ready:
                    self.ready.remove(current_pcb)
                    self.ready.append(current_pcb)
                return
            lvl = current_pcb.queue_level

            # RQ0 = RR -> demote to RQ1 after time slice
            if lvl == 0:
                if current_pcb in self.feedback_queues[0]:
                    self.feedback_queues[0].remove(current_pcb)
                    self.feedback_queues[1].append(current_pcb)
                    current_pcb.queue_level = 1
                    print(f"[SCHED] PID {current_pcb.pid} demoted to RQ1 (HRRN)")
                
            # RQ1 = HRRN -> no demotion
            # RQ2 = FCFS -> no demotion


# Represents the CPU responsible for executing instructions and handling interrupts 
class CPU:
    def __init__(self, mmu, ram, scheduler, pcb_table, metrics):
        self.mmu = mmu
        self.ram = ram
        self.sched = scheduler
        self.pcbs = pcb_table
        self.metrics = metrics 
        
        self.current = None # current running process ID
        self.current_proc_start_tick = 0 # start tick of current process
        self.kernel_mode = False # interrupt handling flag
        self.global_tick = 0 # global simulation clock
        self.time_slice_counter = 0 # Track quantum
        self.IVT = { # interrupt vector table
            'TIMER': self._timer_handler,
            'IO': self._io_handler,
            'SYSCALL': self._syscall_handler,
            'TRAP': self._trap_handler
        }

    # Purpose: Switches CPU control to the next process (Context Switching)
    def dispatch(self, next_pid):
        if self.current is not None:
            oldpcb = self.pcbs[self.current]
            # Log finishing segment for old process with queue label
            self.metrics.log_gantt(
                self.current,
                oldpcb.queue_level,
                self.current_proc_start_tick,
                self.global_tick,
            )
            if oldpcb.state == 'RUNNING':
                oldpcb.state = 'READY'

        # Switch to new process
        self.current = next_pid
        if next_pid is not None:
            pcb = self.pcbs[next_pid]
            pcb.state = 'RUNNING'
            self.time_slice_counter = 0
            self.metrics.context_switches += 1
            self.current_proc_start_tick = self.global_tick
            print(f"[DISPATCH] Context Switch to PID {pcb.pid} (Algo: {self.sched.algorithm})")

    # Purpose: Raises an interrupt and calls its corresponding handler
    def raise_interrupt(self, name, a=None, b=None):
        handler = self.IVT.get(name)
        if handler is not None:
            self.kernel_mode = True
            handler(a, b)
            self.kernel_mode = False

    # Purpose: Decodes an instruction register into opcode and operand
    def decode(self, ir):
        opcode, operand = ir
        return opcode, operand

    # Purpose: Executes one CPU instruction cycle
    def step(self):
        if self.current is None:
            next_pcb = self.sched.select_next(self.global_tick)
            if next_pcb: 
                self.dispatch(next_pcb.pid)
            else: 
                return False 

        pcb = self.pcbs[self.current]
        
        try:
            word = self.mmu.read(pcb.pid, pcb.pc)
        except Exception as e:
            self.raise_interrupt('TRAP', pcb.pid, str(e))
            return False

        pcb.ir = word if isinstance(word, tuple) else (word, None)
        pcb.pc += 1
        opcode, operand = self.decode(pcb.ir)

        did_execute = self._execute_instruction(pcb, opcode, operand)

        # Update metrics and counters
        self.global_tick += 1
        self.metrics.total_ticks += 1

        if did_execute:
            pcb.burst_executed += 1
            self.metrics.cpu_active_ticks += 1
            self.time_slice_counter += 1

            queue_level = pcb.queue_level

            if queue_level == 0 and self.time_slice_counter >= TIME_SLICE_INSTR:
                self.raise_interrupt('TIMER')
                return True
            
            candidate = self.sched.select_next(self.global_tick)
            if candidate and candidate.pid != self.current:
                if candidate.queue_level < queue_level:
                    self.raise_interrupt('TIMER')
        return True
        

    # Purpose: Updates ZF/CF/VF based on operation and result
    def _update_flags(self, pcb, before, opnd, result, op_name):
        pcb.zf = 1 if result == 0 else 0
        MIN32, MAX32 = -2147483648, 2147483647
        pcb.cf = 1 if (result < MIN32 or result > MAX32) else 0
        pcb.vf = 0 

    # Purpose: Executes the instruction currently loaded in IR
    def _execute_instruction(self, pcb, opcode, operand):
        try:
            if opcode == NOP: return True

            if opcode == LOAD:
                val = self.mmu.read(pcb.pid, operand)
                pcb.acc = int(val)

            elif opcode == STORE:
                self.mmu.write(pcb.pid, operand, int(pcb.acc))

            elif opcode == ADD:
                val = self.mmu.read(pcb.pid, operand)
                pcb.acc += int(val)

            elif opcode == SUB:
                val = self.mmu.read(pcb.pid, operand)
                pcb.acc -= int(val)

            elif opcode == IO:
                pcb.state = 'BLOCKED'
                self.sched.add_blocked(pcb)
                self.sched.remove_ready(pcb)
                
                t = threading.Thread(target=self._finish_io_after_delay, args=(pcb.pid,))
                t.daemon = True; t.start()
                
                next_p = self.sched.select_next(self.global_tick)
                if next_p:
                    self.dispatch(next_p.pid)
                else:
                    self.current = None

            elif opcode == SYSCALL:
                self.raise_interrupt('SYSCALL', pcb.pid, operand)

            return True

        except Exception as e:
            self.raise_interrupt('TRAP', pcb.pid, str(e))
            return False

    # Purpose: Completes I/O after a randomized delay
    def _finish_io_after_delay(self, pid):
        time.sleep(random.uniform(IO_MIN, IO_MAX))
        self.raise_interrupt('IO', pid)

    # Purpose: Handles timer interrupt for process switching (RR/Feedback)
    def _timer_handler(self, a=None, b=None):
        if self.current is None: return
        curr_pcb = self.pcbs[self.current]
        
        if self.sched.algorithm in ['RR', 'FEEDBACK']:
            self.sched.rotate_queue(curr_pcb)
        
        next_pcb = self.sched.select_next(self.global_tick)
        if next_pcb and next_pcb.pid != self.current:
            self.dispatch(next_pcb.pid)
        else:
            self.time_slice_counter = 0

    # Purpose: Handles I/O completion interrupt
    def _io_handler(self, pid, b=None):
        if pid in self.pcbs:
            pcb = self.pcbs[pid]
            self.sched.unblock(pcb)
            if self.sched.algorithm in ['PRIORITY', 'SRT']:
                best = self.sched.select_next(self.global_tick)
                if best and best.pid != self.current:
                    self.dispatch(best.pid)

    # Purpose: Handles system call interrupt (Termination)
        # Purpose: Handles system call interrupt (Termination)
    def _syscall_handler(self, pid, code):
        print(f"[INT] SYSCALL PID {pid} Code {code}")
        pcb = self.pcbs[pid]
        # Log final Gantt slice with queue level
        self.metrics.log_gantt(
            pid,
            pcb.queue_level,
            self.current_proc_start_tick,
            self.global_tick
        )
        # Mark as terminated
        pcb.state = 'TERMINATED'
        # Remove from all scheduler structures
        self.sched.remove_ready(pcb)    
        if pcb in self.sched.blocked:
            self.sched.blocked.remove(pcb)
        # Record performance metrics
        self.metrics.log_process(pcb, self.global_tick)
        pcb.pc = 0
        # CPU is now idle; dispatcher will pick the next one on next step()
        self.current = None


    # Purpose: Handles trap or fatal error interrupt
    def _trap_handler(self, pid, reason):
        print(f"[INT] TRAP PID {pid}: {reason}")
        self._syscall_handler(pid, -1)

# Represents the OS Kernel that loads programs and runs the simulation
class Kernel:
    def __init__(self, sched_algo='RR', mem_strategy='FIRST_FIT'):
        self.ram = RAM(RAM_SIZE)
        self.l1 = FIFOCache(L1_SIZE, "L1")
        self.l2 = FIFOCache(L2_SIZE, "L2")
        self.mmu = MMU(self.ram, self.l1, self.l2)
        self.mem_table = MemoryTable()
        
        self.sched_algo = sched_algo
        self.mem_strategy = mem_strategy
        
        self.scheduler = Scheduler(algorithm=sched_algo)
        self.metrics = SimMetrics()
        self.pcbs = {}  # PCB storage
        self.cpu = CPU(self.mmu, self.ram, self.scheduler, self.pcbs, self.metrics)

    # Purpose: Loads a program into memory and initializes its PCB
    def create_process(self, pid, program, priority=0, burst_est=10):
        prog_words = len(program)
        total_words = prog_words + 64 # extra space for data
        
        alloc_start, alloc_end = self.mem_table.allocate(pid, total_words, self.mem_strategy)
        
        num_pages = (total_words + PAGE_SIZE - 1) // PAGE_SIZE
        self.mmu.map_pages(pid, num_pages)
        
        pcb = PCB(pid, burst_estimate=burst_est)
        pcb.state = 'READY'
        pcb.priority = priority
        self.pcbs[pid] = pcb
        self.scheduler.add_ready(pcb)
        
        for i in range(prog_words):
            self.mmu.write(pid, i, program[i])
        
        print(f"[LOAD] PID {pid} loaded. Priority: {priority}, Burst Est: {burst_est}")

    # Purpose: Runs the main simulation loop until all processes terminate
    def run(self, max_steps=2000):
        print(f"Starting Simulation | Algo: {self.sched_algo} | Mem: {self.mem_strategy}")
        steps = 0
        while any(p.state != 'TERMINATED' for p in self.pcbs.values()) and steps < max_steps:
            self.cpu.step()
            steps += 1
            if not self.cpu.current and self.scheduler.blocked:
                time.sleep(0.01) 
        
        self.shutdown()

    # Purpose: Prints a summary of results, frees memory
    def shutdown(self):
        for pid in list(self.pcbs.keys()):
            self.mem_table.deallocate(pid)
        self.metrics.print_report(self.sched_algo)
        self.metrics.export_gantt_chart(self.sched_algo)

# Purpose: Helper to generate instruction lists
def gen_prog(iters, type='CPU'):
    P = [(LOAD, 100)]
    for _ in range(iters):
        P.append((ADD, 100))
        if type == 'IO' and _ % 3 == 0: P.append((IO, None))
    P.append((SYSCALL, 0))
    return P

# Purpose: Automated Test Suite (Module 5)
def run_comparison():
    # Define the workload once so every algorithm runs the same processes
    def get_workload():
        return [
            # PID, Program Instructions, Priority, Burst Estimate
            (1, gen_prog(20, 'CPU'), 1, 20),
            (2, gen_prog(5, 'IO'), 3, 10),
            (3, gen_prog(15, 'CPU'), 2, 15)
        ]

    algorithms = ['FCFS', 'RR', 'SPN', 'SRT', 'HRRN', 'PRIORITY', 'FEEDBACK']
    results = []

    print(f"\nSTARTING COMPARISON SUITE")

    # Print selection menu
    print("\nSelect scheduling algorithm to simulate:")
    for idx, algo in enumerate(algorithms, start=1):
        print(f"{idx}. {algo}")
    all_option = len(algorithms) + 1
    print(f"{all_option}. Run ALL algorithms")

    # Get user choice
    choice = input("Enter your choice (1-8): ")

    try:
        choice = int(choice)
    except ValueError:
        print("Invalid input. Defaulting to running ALL algorithms.")
        selected_algorithms = algorithms
    else:
        if 1 <= choice <= len(algorithms):
            selected_algorithms = [algorithms[choice - 1]]
        elif choice == all_option:
            selected_algorithms = algorithms
        else:
            print("Choice out of range. Defaulting to running ALL algorithms.")
            selected_algorithms = algorithms

    # Run only the selected algorithms
    for algo in selected_algorithms:
        print(f"\nSimulating {algo}...")
        
        k = Kernel(sched_algo=algo, mem_strategy='BEST_FIT')
        
        workload = get_workload()
        for pid, prog, prio, burst in workload:
            k.create_process(pid, prog, priority=prio, burst_est=burst)
            
        k.run()
        
        total_wait = sum(p['waiting'] for p in k.metrics.completed_processes)
        total_turn = sum(p['turnaround'] for p in k.metrics.completed_processes)
        count = len(k.metrics.completed_processes)
        
        avg_wait = total_wait / count if count > 0 else 0
        avg_turn = total_turn / count if count > 0 else 0
        
        results.append({
            'Algorithm': algo,
            'Avg Wait': avg_wait,
            'Avg Turnaround': avg_turn,
            'Context Switches': k.metrics.context_switches
        })

    # Consolidated report only for algorithms that were run
    print(f"\n\nFINAL ALGORITHM COMPARISON")
    print(f"{'Algorithm':<15}{'Avg Wait':<15}{'Avg Turnaround':<18}{'Context Switches':<15}")
    print()
    
    for r in results:
        print(f"{r['Algorithm']:<15}{r['Avg Wait']:<15.4f}{r['Avg Turnaround']:<18.4f}{r['Context Switches']:<15}")
    print()

    try:
        with open("simulation_results.csv", "w") as f:
            f.write("Algorithm,Avg_Wait,Avg_Turnaround,Context_Switches\n")
            for r in results:
                f.write(f"{r['Algorithm']},{r['Avg Wait']},{r['Avg Turnaround']},{r['Context Switches']}\n")
        print("\n[INFO] Results exported to 'simulation_results.csv' for visualization.")
    except:
        print("\n[WARN] Could not save CSV file.")


if __name__ == "__main__":
    run_comparison()
# Project 3 - Operating System Simulation (Advanced Scheduling & Evaluation)

**Course:** CSAS 3111  
**Group:** Emma Trusio, Kyle Collins, Antonio Marin, and Joseph Lodge  
**Professor:** Professor Ashlin Johnsy  
**Date:** December 2025

---

## Overview

This project builds upon **Project 1** and **Project 2** by taking our basic CPU and memory systems and turning them into a working simulation of an operating system capable of multitasking and performance evaluation. It integrates the fundamental concepts of computer systems—like a "rough skeleton" of hardware—with high-level OS management.

It includes everything a real OS would need at an advanced level:
* **CPU:** Fetches and executes instructions using a defined instruction set.
* **MMU:** Handles caching, paging, and virtual-to-physical address translation.
* **Advanced Scheduling:** Implements multiple algorithms to handle CPU responses to multiple tasks at once.
* **Interrupts & I/O Handling:** Allows the system to multitask by pausing events to handle internal or external triggers.

---

## System Breakdown

### 1. CPU and Instruction Cycle
The CPU runs through a **fetch-decode-execute** cycle via the `step()` function.
* **Fetch:** The CPU receives instructions from memory, transferring the address in the PC to the MAR and placing the instruction into the MDR and IR.
* **Decode:** The Control Unit checks the opcode to determine the operation and identify operands.
* **Execute:** The CPU performs the actual operation determined during the decoding stage.

The system tracks several key registers and flags:
* **PC (Program Counter):** Points to the next instruction's address.
* **ACC (Accumulator):** Stores arithmetic results.
* **IR (Instruction Register):** Holds the fetched instruction.
* **Flags:** 'ZF' (Zero), 'CF' (Carry), and 'VF' (Overflow) track the condition of operations.

### 2. Memory System
The system utilizes a multi-layered hierarchy to manage data efficiently:
* **RAM:** Acts as main memory for storing data and instructions.
* **Cache (L1 & L2):** Uses a **FIFOCache** (First-In-First-Out) replacement policy to optimize access time based on size vs. speed tradeoffs.
* **MMU (Memory Management Unit):** Performs address translation using Page Tables to map virtual pages to physical frames.
* **MemoryTable:** Handles memory allocation (First-Fit/Best-Fit) and deallocation.
    * **At process creation:** `[MEM] Allocation` logs show the assigned address range.
    * **During runtime:** `[MEMORY TABLE]` snapshots display used and free regions.
    * **Upon termination:** `[MEM] Free` logs indicate released memory, followed by the merging of adjacent free blocks (compaction).

### 3. Process Scheduling
Each process is represented by a **PCB (Process Control Block)** containing its state, registers, and timing data. The **Scheduler** manages ready and blocked queues to coordinate execution. Supported algorithms include:
* **FCFS:** First-Come, First-Served; susceptible to the "convoy effect".
* **RR:** Round Robin; rotates processes based on a fixed time slice (5 instructions).
* **HRRN:** Highest Response Ratio Next; evaluates priority based on wait time and service time.
* **SRT/SPN:** Prioritizes jobs with the shortest remaining or total execution time.
* **Feedback:** Uses multi-level queues to demote CPU-intensive processes from RR to HRRN or FCFS.



### 4. Interrupt Handling
The CPU uses an **Interrupt Vector Table (IVT)** to manage system events:
* **TIMER:** Triggered when a process's time slice expires.
* **IO:** Triggered when a simulated I/O operation completes.
* **SYSCALL:** Used for process termination or OS service requests.
* **TRAP:** Handles errors such as page faults or invalid operations.

### 5. I/O and Multithreading
I/O operations allow the system to multitask by running multiple threads on a single process.
* When an **IO** opcode is executed, the process is marked as **BLOCKED** and moved to the blocked queue.
* A separate Python thread starts a randomized delay (0.05 to 0.10 seconds).
* Once the delay ends, an **IO interrupt** is raised to unblock the process and return it to the ready queue.

---

## Output Logs and Evaluation
The simulator performs a quantitative comparison of algorithms based on several metrics:
* **Average Waiting Time:** Time spent in the ready queue.
* **Average Turnaround Time (TAT):** Total time from arrival to completion.
* **CPU Utilization:** Percentage of time the CPU was active.
* **Context Switches:** Number of times the CPU switched between processes.

---

## How to Run - From Working Directory
1. Open the project in your IDE.
2. Run the program:  
   `python3 Final_Project_CSAS3111.py` 
3. Select a scheduling algorithm (1-7) or simulate all when prompted.
4. View the results in the terminal or the exported `simulation_results.csv`.

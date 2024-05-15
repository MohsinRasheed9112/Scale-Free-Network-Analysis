import hashlib
import datetime as date
import random
import time
import threading
import statistics
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Set a fixed seed for reproducibility
random.seed(42)

class Transaction:
    def __init__(self, sender, recipient, amount, complexity=1, signature=None):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount
        self.complexity = complexity
        self.signature = signature

    def to_dict(self):
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "signature": self.signature,
        }

    def sign(self, signature):
        self.signature = signature

class Block:
    def __init__(self, index, timestamp, transactions, previous_hash, nonce=0, network_node=None):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.network_node = network_node
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_header = str(self.index) + str(self.timestamp) + str(self.previous_hash) + str(self.nonce) + str(self.network_node)
        block_transactions = "".join([str(tx.to_dict()) for tx in self.transactions])
        hash_string = block_header + block_transactions
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def mine_block(self, difficulty):
        target = '0' * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        print("Block mined: ", self.hash)  # Commented out to reduce console clutter

class SimpleBlockchain:
    def __init__(self, difficulty=2):
        self.chain = [self.create_genesis_block()]
        self.difficulty = difficulty
        self.pending_transactions = []
        self.mining_reward = 10

    def create_genesis_block(self):
        return Block(0, date.datetime.now(), [], "0", network_node=None)

    def mine_pending_transactions(self, mining_reward_address):
        print("Starting to mine pending transactions...")
        if not self.pending_transactions:
            print("No transactions to mine.")
            return

        valid_transactions = [tx for tx in self.pending_transactions if self.validate_transaction(tx)]
        if not valid_transactions:
            print("No valid transactions to mine.")
            return

        print(f"Mining a block with {len(valid_transactions)} transactions.")
        block = Block(len(self.chain), date.datetime.now(), valid_transactions, self.get_latest_block().hash)
        block.mine_block(self.difficulty)

        print(f"Block successfully mined! Hash: {block.hash}")
        self.chain.append(block)
        self.pending_transactions = [Transaction(None, mining_reward_address, self.mining_reward)]

    def get_latest_block(self):
        return self.chain[-1]

    def create_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def validate_transaction(self, transaction):
        if transaction.amount <= 0:
            return False
        if not transaction.sender or not transaction.recipient:
            return False
        return True

    def is_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            if current_block.hash != current_block.calculate_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
        return True
    

class ScaleFreeNetwork:
    def __init__(self, initial_nodes=3, m=5):
        self.nodes = list(range(initial_nodes))
        self.edges = [(i, j) for i in self.nodes for j in self.nodes if i != j]
        self.m = m
        self.dynamic_m_adjustment_enabled = True

    def node_degree(self, node):
        return sum(1 for edge in self.edges if node in edge)

    def adjust_m_based_on_network_conditions(self):
        print("Adjusting 'm' based on network conditions...")
        network_size = len(self.nodes)
        average_degree = sum(self.node_degree(node) for node in self.nodes) / network_size
        target_degree = 8  # Example target connectivity

        if average_degree < target_degree:
            self.m += 1
        elif average_degree > target_degree and self.m > 1:
            self.m -= 1
        # Ensure 'm' stays within practical limits
        self.m = max(1, min(self.m, 10))

        # Print the current value of 'm' after adjustment
        print(f"Value of 'm' adjusted to: {self.m}")

    def add_node(self):
        new_node_id = len(self.nodes)
        self.nodes.append(new_node_id)

        potential_edges = [(new_node_id, node) for node in self.nodes if node != new_node_id]
        chosen_edges = random.choices(potential_edges, weights=[self.node_degree(node) for node in self.nodes if node != new_node_id], k=self.m)
        self.edges.extend(chosen_edges)
        return new_node_id

    def elect_leaders(self):
        degrees = {node: 0 for node in self.nodes}
        for edge in self.edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
        return sorted_nodes[:max(3, len(self.nodes) // 10)]

    def adjust_m_dynamically(self, target_throughput, actual_throughput):
        # Simplified to focus on dynamic adjustment logic
        if actual_throughput < target_throughput:
            self.m = min(self.m + 1, 10)
        else:
            self.m = max(self.m - 1, 1)
        print(f"Dynamically adjusted 'm' to: {self.m}")

class Blockchain(SimpleBlockchain):
    def __init__(self, difficulty=2):
        super().__init__(difficulty)
        self.network = ScaleFreeNetwork()
        self.adapt_difficulty()  # Adapt initial difficulty based on network conditions
    
    def adapt_difficulty(self):
        self.difficulty = max(2, min(self.difficulty, len(self.network.nodes) // 10))
        print(f"Difficulty adjusted to {self.difficulty}.")

    def mine_pending_transactions(self, mining_reward_address):
        start_time = time.time()
        super().mine_pending_transactions(mining_reward_address)
        end_time = time.time()
        print(f"Mining took {end_time - start_time} seconds.")

    def validate_block_with_leaders(self, block):
        # Leader election and block validation
        leaders = self.network.elect_leaders()
        return any(leader == block.network_node for leader in leaders)

    def mine_in_parallel(self, mining_reward_address):
        # Parallel block mining
        new_node_id = self.network.add_node()
        leaders = self.network.elect_leaders()
        if new_node_id in leaders:
            self.mine_pending_transactions(mining_reward_address)
            self.chain[-1].network_node = new_node_id

def perform_measurements(blockchain, num_transactions=100):
    throughput_times = []
    for _ in range(num_transactions):
        complexity = 1  # Fixed transaction complexity
        transaction = Transaction("Sender", "Recipient", random.randint(1, 100), complexity)
        start_time = time.perf_counter()
        blockchain.create_transaction(transaction)
        blockchain.mine_pending_transactions("Miner")
        end_time = time.perf_counter()
        adjusted_time = end_time - start_time
        throughput_times.append(adjusted_time)
    throughput = num_transactions / sum(throughput_times)
    latency = sum(throughput_times) / num_transactions
    return throughput, latency


def simulate_performance_for_m_values(m_values, network_sizes, runs_per_config=5):
    results = {}
    for m in m_values:
        for size in network_sizes:
            throughputs, latencies, leader_counts = [], [], []
            for _ in range(runs_per_config):
                blockchain = Blockchain(difficulty=2)
                blockchain.network = ScaleFreeNetwork(initial_nodes=3, m=m)
                while len(blockchain.network.nodes) < size:
                    blockchain.network.add_node()
                throughput, latency = perform_measurements(blockchain, num_transactions=100)
                leader_count = len(blockchain.network.elect_leaders())
                throughputs.append(throughput)
                latencies.append(latency)
                leader_counts.append(leader_count)
            results[(m, size)] = (statistics.mean(throughputs), statistics.stdev(throughputs),
                                  statistics.mean(latencies), statistics.stdev(latencies),
                                  statistics.mean(leader_counts), statistics.stdev(leader_counts))
    return results

def print_results(results):
    # Assuming 'results' is a dictionary with keys as (m, size) tuples and values as performance metrics
    for (m, size), metrics in results.items():
        avg_throughput, std_throughput, avg_latency, std_latency, avg_leader_count, std_leader_count = metrics
        print(f"Configuration: m={m}, Network Size={size}")
        print(f"  Avg Throughput: {avg_throughput:.2f} tps (±{std_throughput:.2f})")
        print(f"  Avg Latency: {avg_latency:.4f} s (±{std_latency:.4f})")
        print(f"  Avg Leader Count: {avg_leader_count:.2f} (±{std_leader_count:.2f})\n")

def visualize_results(results):
    plt.figure(figsize=(14, 7))
    
    # Throughput plot
    plt.subplot(1, 2, 1)
    for size in network_sizes:
        #m_values = range(1, 11)
        throughputs = [results[(m, size)][0] for m in m_values]
        plt.plot(m_values, throughputs, label=f'Network Size {size}')
    plt.xlabel('m Value')
    plt.ylabel('Average Throughput (tps)')
    plt.title('Throughput vs. m Value')
    plt.legend()

    # Latency plot
    plt.subplot(1, 2, 2)
    for size in network_sizes:
        #m_values = range(1, 11)
        latencies = [results[(m, size)][2] for m in m_values]  # 2 for average latency index
        plt.plot(m_values, latencies, label=f'Network Size {size}')
    plt.xlabel('m Value')
    plt.ylabel('Average Latency (s)')
    plt.title('Latency vs. m Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

m_values = range(1, 21)
network_sizes = [25, 50, 100, 200, 300]  # Defined globally if used in multiple functions
results = simulate_performance_for_m_values(m_values, network_sizes)

print_results(results)
visualize_results(results)
import datetime
import hashlib
import json

class Blockchain:
    """A simple class for creating a proof-of-work blockchain for logging."""
    
    def __init__(self):
        self.chain = []
        # Create the genesis block (the very first block in the chain)
        self.create_block(proof=1, previous_hash='0', data='Genesis Block: System Initialized')

    def create_block(self, proof, previous_hash, data):
        """Creates a new block and adds it to the chain."""
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(datetime.datetime.now()),
            'data': data,
            'proof': proof,
            'previous_hash': previous_hash
        }
        self.chain.append(block)
        return block

    def get_previous_block(self):
        """Returns the last block in the chain."""
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        """
        A simple consensus algorithm (mining puzzle).
        Find a 'proof' number that results in a hash with 4 leading zeros.
        """
        new_proof = 1
        while True:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                return new_proof
            new_proof += 1

    def hash(self, block):
        """Creates a SHA-256 hash of a given block."""
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def add_event(self, event_data):
        """
        A high-level function to add a new event to the blockchain.
        It handles the proof of work and chaining automatically.
        """
        previous_block = self.get_previous_block()
        previous_proof = previous_block['proof']
        proof = self.proof_of_work(previous_proof)
        previous_hash = self.hash(previous_block)
        self.create_block(proof, previous_hash, data=event_data)
        print(f"Blockchain event added: {event_data}")


# blockchain.py

import datetime
import hashlib
import json

class Blockchain:
    def __init__(self):
        self.chain = []
        # The first block is called the "genesis block"
        self.create_block(proof=1, previous_hash='0', data='Genesis Block')

    def create_block(self, proof, previous_hash, data):
        """
        Creates a new block in the blockchain.
        `data` can be any string (e.g., file access log, user ID).
        """
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
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        """
        Simple proof-of-work: find a number such that the hash has four leading zeros.
        """
        new_proof = 1
        while True:
            hash_val = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_val[:4] == '0000':
                return new_proof
            new_proof += 1

    def hash(self, block):
        """
        Creates a SHA-256 hash of a block.
        """
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def is_chain_valid(self):
        """
        Checks chain validity (no tampering).
        """
        previous_block = self.chain[0]
        for index in range(1, len(self.chain)):
            block = self.chain[index]
            # Check hash links
            if block['previous_hash'] != self.hash(previous_block):
                return False
            # Check proof of work
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_val = hashlib.sha256(str(proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_val[:4] != '0000':
                return False
            previous_block = block
        return True

# Example Usage:
if __name__ == "__main__":
    blockchain = Blockchain()
    # Log a file access event
    event_data = "User A accessed file secrets.txt"
    prev_block = blockchain.get_previous_block()
    proof = blockchain.proof_of_work(prev_block['proof'])
    prev_hash = blockchain.hash(prev_block)
    new_block = blockchain.create_block(proof, prev_hash, data=event_data)
    print("New block added:")
    print(json.dumps(new_block, indent=2))
    print("Blockchain valid?", blockchain.is_chain_valid())
    print("Full chain:")
    print(json.dumps(blockchain.chain, indent=2))

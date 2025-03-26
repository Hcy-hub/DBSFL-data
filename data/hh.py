import heapq
from collections import defaultdict

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    char_freq = defaultdict(int)
    for item in data:
        char_freq[item] += 1

    heap = [Node(char, freq) for char, freq in char_freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def build_huffman_codes(node, current_code, huffman_codes):
    if node is None:
        return

    if node.char:
        huffman_codes[node.char] = current_code
        return

    build_huffman_codes(node.left, current_code + '0', huffman_codes)
    build_huffman_codes(node.right, current_code + '1', huffman_codes)

def huffman_encode(data):
    root = build_huffman_tree(data)
    huffman_codes = {}
    build_huffman_codes(root, '', huffman_codes)

    encoded_data = [huffman_codes[item] for item in data]
    return encoded_data, root

def huffman_decode(encoded_data, root):
    decoded_data = []
    current_node = root

    for bits in encoded_data:
        for bit in bits:
            if bit == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right

            if current_node.char:
                decoded_data.append(current_node.char)
                current_node = root

    return decoded_data

if __name__ == '__main__':
    data = [1, 2, 1, 3, 3, 2, 1, 1, 2]
    encoded_data, huffman_tree = huffman_encode(data)
    print(f"Encoded data: {encoded_data}")

    decoded_data = huffman_decode(encoded_data, huffman_tree)
    print(f"Decoded data: {decoded_data}")

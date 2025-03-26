'''
    huffman编码
'''
import copy

class Node:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight
        self.left = None
        self.right = None
        self.father = None

    def is_left_child(self):
        return self.father.left == self


def create_prim_nodes(data_set, labels):
    if(len(data_set) != len(labels)):
        raise Exception('')
    nodes = []
    for i in range(len(labels)):
        nodes.append( Node(labels[i],data_set[i]) )
    return nodes


# 创建huffman树
def create_HF_tree(nodes):

    tree_nodes = nodes.copy()
    while len(tree_nodes) > 1:
        tree_nodes.sort(key=lambda node: node.weight)
        new_left = tree_nodes.pop(0)
        new_right = tree_nodes.pop(0)
        new_node = Node(None, (new_left.weight + new_right.weight))
        new_node.left = new_left
        new_node.right = new_right
        new_left.father = new_right.father = new_node
        tree_nodes.append(new_node)
    tree_nodes[0].father = None
    return tree_nodes[0]

#获取huffman编码
def get_huffman_code(nodes):
    codes = {}
    for node in nodes:
        code=''
        name = node.name
        while node.father != None:
            if node.is_left_child():
                code = '0' + code
            else:
                code = '1' + code
            node = node.father
        codes[name] = code
    return codes

def huffman_decode(encode_data,root):
    decode_data = ''
    current_node = root

    for bit in encode_data:
        if bit == '0':
            current_node = current_node.left
        elif bit == '1':
            current_node = current_node.right
        if current_node.left is None and current_node.right is None:
            decode_data += current_node.name
            current_node = root
        return decode_data

if __name__ == '__main__':
    labels = ['a','b','c','d','e','f']
    data_set = [9,12,6,3,5,15]
    nodes = create_prim_nodes(data_set,labels)
    root = create_HF_tree(nodes)
    codes = get_huffman_code(nodes)

    for key in codes.keys():
        print(key,': ',codes[key])
    results = ''.join(codes.values())
    print(results)
    decoded_data = huffman_decode(results, root)
    print("Decoded Data:", decoded_data)

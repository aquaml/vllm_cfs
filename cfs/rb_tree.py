from enum import Enum

class rb_node_color(Enum):
    RED = 1
    BLACK = 2

class rb_node:
    def __init__(self, container) -> None:
        self.left: rb_node = None
        self.right: rb_node = None
        self.parent: rb_node = None
        self.container = container
        self.color: rb_node_color = None
        
    # Override this method to keep other objects in the tree
    def get_value(self) -> int:
        return self.container.get_value()

class rb_tree:

    def __init__(self, null_node: rb_node) -> None:
        self.num_elements: int = 0
        self.tnull = null_node
        self.tnull.color = rb_node_color.BLACK
        self.root: rb_node = null_node
    
    def insert(self, node: rb_node):
        self.num_elements += 1
        node.left = self.tnull
        node.right = self.tnull

        if self.root == self.tnull:
            self.root = node
            self.root.color = rb_node_color.BLACK
            return

        current: rb_node = self.root
        node.color = rb_node_color.RED
        parent: rb_node = None

        while current != self.tnull:
            parent = current
            if node.get_value() >= current.get_value():
                current = current.right
            else:
                current = current.left
        
        node.parent = parent
        if node.get_value() >= parent.get_value():
            parent.right = node
        else:
            parent.left = node
        
        if node.parent.parent == None:
            return
        
        # print("Fixing insertion of: {}".format(node.get_value()))
        # self.print_tree()
        # print("=========")
        self.adjust_rb_properties_after_insertion(node)

    def left_rotate(self, node: rb_node):
        if node.parent:
            if node.parent.left == node:
                node.parent.left = node.right
            else:
                node.parent.right = node.right

        node.right.parent = node.parent
        right_child = node.right

        node.right = right_child.left
        if node.right != self.tnull:
            node.right.parent = node
        
        right_child.left = node
        node.parent = right_child

        if right_child.parent == None:
            self.root = right_child

    def right_rotate(self, node: rb_node):
        if node.parent:
            if node.parent.left == node:
                node.parent.left = node.left
            else:
                node.parent.right = node.left
        
        node.left.parent = node.parent
        left_child = node.left

        node.left = left_child.right
        if node.left != self.tnull:
            node.left.parent = node
        
        left_child.right = node
        node.parent = left_child

        if left_child.parent == None:
            self.root = left_child
    
    def get_least(self) -> rb_node:
        current = self.root
        if current == None:
            return current

        while current.left != self.tnull:
            current = current.left
       
        assert current.left == self.tnull
        is_black_node = current.color == rb_node_color.BLACK
        if current.left == self.tnull:
            fix_begin_location = current.right
            self.transplant(current, current.right)
        # We always return the last left node, so, no need to do binary tree transplant
        # print("Adjusting properties for node: {}".format(current.get_value()))
        if is_black_node:
            self.adjust_rb_properties_after_deletion(fix_begin_location)
        return current

    def transplant(self, u, v):
        if u.parent == None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def adjust_rb_properties_after_deletion(self, node: rb_node):
        while node != self.root and node.color == rb_node_color.BLACK:
            # If node is parent's left child
            # print("Node is : {}".format(node.get_value()))
            if node == node.parent.left:
                sibling = node.parent.right
                # If sibling is red
                if sibling.color == rb_node_color.RED:
                    # Change sibling's color to black
                    sibling.color = rb_node_color.BLACK
                    # Change parent's color to red
                    sibling.parent.color = rb_node_color.RED
                    # Left rotate parent 
                    # print("LR 3: {}".format(sibling.parent.get_value()))
                    # self.print_tree()
                    self.left_rotate(sibling.parent)
                    # self.print_tree()
                    # sibling would have changed, re-assign
                    sibling = node.parent.right
                # If both the sibling's children are black
                if sibling.left.color == rb_node_color.BLACK and sibling.right.color == rb_node_color.BLACK:
                    sibling.color = rb_node_color.RED
                    node = node.parent
                # one of the children are red
                else:
                    # Left child is red
                    if sibling.right.color == rb_node_color.BLACK:
                        if sibling.left:
                            sibling.left.color = rb_node_color.BLACK
                        sibling.color = rb_node_color.RED
                        # print("RR 3: {}".format(sibling.get_value()))
                        # self.print_tree()
                        self.right_rotate(sibling)
                        # self.print_tree()
                        sibling = node.parent.right
                    # Right is red or left is red
                    sibling.color = node.parent.color
                    node.parent.color = rb_node_color.BLACK
                    sibling.right.color = rb_node_color.BLACK
                    # print("LR 3.1: {}".format(node.parent.get_value()))
                    # self.print_tree()
                    self.left_rotate(node.parent)
                    # self.print_tree()
                    node = self.root
            else:
                sibling = node.parent.left
                if sibling and sibling.color == rb_node_color.RED:
                    sibling.color = rb_node_color.BLACK
                    node.parent.color = rb_node_color.RED
                    # print("RR 4")
                    # self.print_tree()
                    self.right_rotate(sibling.parent)
                    # self.print_tree()
                    sibling = node.parent.left

                if sibling.right.color == rb_node_color.BLACK and sibling.right.color == rb_node_color.BLACK:
                    sibling.color = rb_node_color.RED
                    node = node.parent
                else:
                    if sibling.left.color == rb_node_color.BLACK:
                        if sibling.right:
                            sibling.right.color = rb_node_color.BLACK
                        sibling.color = rb_node_color.RED
                        # print("LR 4")
                        # self.print_tree()
                        self.left_rotate(sibling)
                        # self.print_tree()
                        sibling = node.parent.left
                    if sibling:
                        sibling.color = node.parent.color
                    node.parent.color = rb_node_color.BLACK
                    if sibling and sibling.left:
                        sibling.left.color = rb_node_color.BLACK
                    # print("RR 4.1 at node parent value {}, node value: {}".format(node.parent.get_value(), node.get_value()))
                    # self.print_tree()
                    self.right_rotate(node.parent)
                    # self.print_tree()
                    node = self.root

        node.color = rb_node_color.BLACK

    def adjust_rb_properties_after_insertion(self, new_node: rb_node):
        while new_node != None and new_node.parent != None and new_node.parent.color == rb_node_color.RED:
            if new_node.parent.parent.right == new_node.parent:
                # parent is the right child
                # parent's sibling 
                parent_sibling = new_node.parent.parent.left
                # If parent sibling is red
                if parent_sibling != None and parent_sibling.color == rb_node_color.RED:
                    parent_sibling.color = rb_node_color.BLACK
                    new_node.parent.color = rb_node_color.BLACK
                    new_node.parent.parent.color = rb_node_color.RED
                    new_node = new_node.parent.parent
                # Else if parent sibling is absent or black
                else:
                    # If new node is the left child of parent, need a right rotation on parent
                    if new_node == new_node.parent.left:
                        new_node = new_node.parent
                        # self.print_tree()
                        # print("RR - 1: {}".format(new_node.get_value()))
                        self.right_rotate(new_node)
                        # self.print_tree()

                    # Now new node is in between, so left rotate its parent
                    new_node.parent.color = rb_node_color.BLACK
                    new_node.color = rb_node_color.RED
                    new_node.parent.parent.color = rb_node_color.RED
                    # self.print_tree()
                    # print("LL - 1: {}".format(new_node.parent.parent.get_value()))
                    self.left_rotate(new_node.parent.parent)
                    # self.print_tree()
                    # Doesn't matter what the value of new_node is now, because the parent is black
                    new_node = new_node.right
            else:
                # parent is the left child
                # parent's sibling
                parent_sibling = new_node.parent.parent.right
                # If parent sibling is red
                if parent_sibling != None and parent_sibling.color == rb_node_color.RED:
                    parent_sibling.color = rb_node_color.BLACK
                    new_node.parent.color = rb_node_color.BLACK
                    new_node.parent.parent.color = rb_node_color.RED
                    new_node = new_node.parent.parent
                # Else if parent sibling is absent or black
                else:
                    # If new node is the right child of the parent, need a left rotation on parent
                    if new_node == new_node.parent.right:
                        new_node = new_node.parent
                        # self.print_tree()
                        # print("LL - 2: {}".format(new_node.get_value()))
                        self.left_rotate(new_node)
                        # self.print_tree()
                    # Now new node is in between, so right rotate its parent
                    new_node.parent.color = rb_node_color.BLACK
                    new_node.color = rb_node_color.RED
                    new_node.parent.parent.color = rb_node_color.RED
                    # self.print_tree()
                    # print("RR - 2: {}".format(new_node.parent.parent.get_value()))
                    self.right_rotate(new_node.parent.parent)
                    # self.print_tree()

                    # Just mark new_node as one of the children to end the loop
                    new_node = new_node.right
            
            self.root.color = rb_node_color.BLACK
    
    def print_tree_helper(self, node: rb_node, depth: int, width: int, print_dict):
        if node == None or node == self.tnull:
            return
        if depth not in print_dict:
            print_dict[depth] = []
        print_str = " " + str(node.get_value()) + " "
        print_str += "R " if node.color == rb_node_color.RED else "B "
        if node.parent:
            print_str += " (Cof): {}".format(node.parent.get_value())
        print_dict[depth].append(print_str)
        self.print_tree_helper(node.left, depth + 1, width - 1, print_dict)
        self.print_tree_helper(node.right, depth + 1, width + 1, print_dict)

    def print_tree(self):
        print_dict = {}
        self.print_tree_helper(self.root, 1, 0, print_dict)
        depth_keys = list(print_dict.keys())
        depth_keys.sort()
        for depth in depth_keys:
            print(print_dict[depth])


def insert_test(nums):
    for num in nums:
        container = int_container(num)
        node = rb_node(container)
        red_black_tree.insert(node)

def delete_test(nums):
    for _ in range(len(nums)):
        test = red_black_tree.get_least()
        print("least value: {}".format(test.get_value()))
        red_black_tree.print_tree()

import random
# random.seed(1)

nums = [0, 1, 2, 3, 4, 5, 6, 7, 8]
random.shuffle(nums)
# nums = [4, 8, 3, 7, 6, 5, 1, 2]
# nums = [1, 2, 3, 4, 5, 6, 7]
# nums.reverse()
print(nums)


class int_container:
    def __init__(self, val: int) -> None:
        self.value = val
    
    def get_value(self) -> int:
        return self.value

red_black_tree = rb_tree(rb_node(int_container(-1)))


for i in range(300):
    print("---------------------")
    random.shuffle(nums)
    insert_test(nums)
    red_black_tree.print_tree()
    delete_test(nums)
    red_black_tree.print_tree()


class TreeNode:

	def __init__(self, node_name, value):
		self.node_name = node_name
		self.value = value
		self.count = 1
		self.leaves = {}

	def add_child(self, child_node):
		self.leaves[child_node.node_name] = child_node

	def get_value(self):
		return self.value

	def update_value(self, new_value):
		self.value = new_value

	def check_child_exist(self, child_name):
		return child_name in self.leaves

	def get_child(self, child_name):
		if self.check_child_exist(child_name):
			return self.leaves[child_name]
		else:
			return None

	def add_count(self):
		self.count += 1

	def get_count(self):
		return self.count


def getExperienceTree():

	root = TreeNode('root', None)
	return root
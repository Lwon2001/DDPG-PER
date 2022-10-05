import numpy


# SumTree

class SumTree(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)  # capacity即为叶子节点数目，2 * capacity - 1即为sumtree中的节点总数，保存优先级数据
        self.data = numpy.zeros(capacity, dtype=object)  # 保存transition数据
        self.write = 0  # 下一个transitino所被保存的指针
        self.n_entries = 0  # 加入到树中的transition数量（包括以后被舍弃的）

    # 加入新节点后父节点的data值发生变化，需要进行修改（向上传播）
    def propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent.any() != 0:
            self.propagate(parent, change)

    # 根据s，找到对应的数据的idx（对应tree）
    def retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def get_n_entries(self):
        return self.n_entries

    # 存储transition及对应的优先级
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update_priority(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # 更新优先级
    def update_priority(self, idx, p):
        idx = numpy.array(idx)
        p = p.flatten()  # 统一维度
        change = p - self.tree[idx]  # 改变量

        self.tree[idx] = p  # 更新优先级
        self.propagate(idx, change)  # 维护sumtree（向上传播改变量）

    # 从sumtree中取出transition及对应的优先级
    def get(self, s):
        idx = self.retrieve(0, s)
        dataIdx = idx - self.capacity + 1  # tree中的idx 减去 (capacity - 1)后得到在data中的dataIdx
        return idx, self.tree[idx], self.data[dataIdx]

"""There are common utilities for all library
"""
import heapq


def levenshtein(s: str, t: str):
    """Levenshtein distance between two strings

    author:

    Christopher P. Matthews,
    christophermatthews1985@gmail.com,
    Sacramento, CA, USA
    """
    if s == t:
        return 0
    elif not s:
        return len(t)
    elif not t:
        return len(s)

    v0 = [i for i in range(len(t) + 1)]
    v1 = v0.copy()
    # pylint: disable=consider-using-enumerate
    for i in range(len(s)):
        v1[0] = i + 1
        # pylint: disable=consider-using-enumerate
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0 = v1[:len(v0)]
    return v1[len(t)]


class UniquePriorityQueue:
    """Priority queue containing only unique values"""
    def __init__(self):
        self.heap = []

    def add(self, d):
        if d not in self.heap:
            heapq.heappush(self.heap, d)

    def to_list(self):
        l = []
        while self.heap:
            l.append(heapq.heappop(self.heap))
        return l


class KeyMapMixin:
    """Mixin that processes key maps"""
    def __init__(self):
        self.key_map = None

    def init_key_map(self, key_map: dict):
        self.key_map = key_map.copy()

        for k, v in key_map.items():
            if not isinstance(k, str):
                raise TypeError(f"Key name '{k}' must be instance of str")
            if not isinstance(v[0], tuple) and not isinstance(v[0], list):
                raise TypeError(f"Real keys '{v[0]}' must be located in list or tuple container")
            if len(v) > 1:
                if not isinstance(v[1], tuple) and not isinstance(v[1], list):
                    raise TypeError(f"Value of preprocessors for '{k}' must be located in list or tuple container")
                self.key_map[k] = (v[0], v[1])
            else:
                self.key_map[k] = (v[0], None)
        return self.key_map

    def retrieve(self, key, obj):
        value = obj
        for v in self.key_map[key][0]:
            value = value[v]
        if self.key_map[key][1]:
            for processor in self.key_map[key][1]:
                value = processor(value)
        return value

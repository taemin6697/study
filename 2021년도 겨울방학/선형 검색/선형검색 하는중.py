from typing import Any,Type
import hashlib

class Node:#노드를 생성해준다.
    def __init__(self,key:Any,value:Any,next:Node)->None:
        self.key = key
        self.value = value
        self.netx = next


class ChainedHash:#체인 해시 정의 
    def __init__(self,capacity:int)->None:
        self.capacity = capacity
        self.table = [None] * self.capacity
        
    def hash_value(self,key:Any)->int:
        if isinstance(key, int):
            return key% self.capacity
        return (int(hashlib.sha256(str(key).encode()).hexdigest(),16)%self.capacity)

    def search(self,key:Any)->Any:
        hash = self.hash_value(key)
        p = self.table[hash]

        while p is not None:
            if p.key == key:
                return p.value
            p = p.next
        return None
    

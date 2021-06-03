import sys
class keras_layer():
    name=None
    kind=None
    def __init__(self,name,kind) -> None:
        super().__init__()
        self.name=name
        self.kind=kind
        
    def __str__(self) -> str:
        return "name={},kind={}".format(self.name,self.kind)

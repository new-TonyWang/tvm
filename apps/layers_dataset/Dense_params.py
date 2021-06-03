from .basic_params import _Global_var_dict

def get_unit():
    current = 2
      def get_next():
        nonlocal ptr
        if(ptr<len(activtions)):
            result = activtions[ptr]
            ptr = ptr+1
            return result
        else:
            ptr = ptr%len(activtions)
            return activtions[ptr]

    return get_next()

_Dense_var_dict={
    "unit":1
}

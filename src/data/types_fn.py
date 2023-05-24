def float_to_str(float, separator):
    return f"{str(float).replace('.', separator)}"

def tuple_to_str(tuple):
    string = f"{'-'.join([float_to_str(elem, ',') for elem in tuple])}"

    return string

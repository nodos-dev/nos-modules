import nodos as nos
import struct

def execute_node(args: nos.NodeExecuteArgs):
    base_mem: memoryview = args.pin_value("Base")
    exp_mem: memoryview = args.pin_value("Exponent")
    out_mem: memoryview = args.pin_value("Result")
    base = struct.unpack('f', base_mem.cast('B'))[0]
    exp = struct.unpack('f', exp_mem.cast('B'))[0]
    # You can also directly set the memory if your type does not 
    # need custom handling & you are inside Nodos Scheduler thread.
    # struct.pack_into('f', out_mem, 0, base ** exp)
    result_buf = struct.pack('f', base ** exp)
    nos.set_pin_value(args.pin_id("Result"), result_buf)
    return nos.result.SUCCESS
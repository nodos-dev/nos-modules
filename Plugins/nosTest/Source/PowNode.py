import struct
import nodos as nosInternal
from typing import Any

class PyPow(nosInternal.nosNode):
    def __init__(self):
        self.previous_base = 0
        self.previous_exponent = 0

    def on_pin_value_changed(self, args: nosInternal.PinValueChangedArgs):
        base_mem: memoryview = args.pin_value

        if(args.pin_name == "Base"):
            nosInternal.log_info(f"Pin {args.pin_name} changed from {self.previous_base} to {struct.unpack('f', base_mem.cast('B'))[0]}")
            self.previous_base = struct.unpack('f', base_mem.cast('B'))[0]
        elif(args.pin_name == "Exponent"):
            nosInternal.log_info(f"Pin {args.pin_name} changed from {self.previous_exponent} to {struct.unpack('f', base_mem.cast('B'))[0]}")
            self.previous_exponent = struct.unpack('f', base_mem.cast('B'))[0]

    def execute_node(self, args: nosInternal.NodeExecuteArgs):
        base_mem: memoryview = args.get_pin_value("Base")
        exp_mem: memoryview = args.get_pin_value("Exponent")
        base = struct.unpack('f', base_mem.cast('B'))[0]
        exp = struct.unpack('f', exp_mem.cast('B'))[0]
        # You can also directly set the memory if your type does not 
        # need custom handling & you are inside Nodos Scheduler thread.
        # struct.pack_into('f', out_mem, 0, base ** exp)
        result_buf = struct.pack('f', base ** exp)
        nosInternal.set_pin_value(args.get_pin_id("Result"), result_buf)
        return nosInternal.result.SUCCESS
    
    def on_node_created(self, args: Any):
        print("Node created")
    
    def on_pin_connected(self, args: nosInternal.PinConnectedArgs):
        print("Pin connected")
    
    def on_pin_disconnected(self, args: nosInternal.PinDisconnectedArgs):
        print("Pin disconnected")